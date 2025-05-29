import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl
import spectrochempy as scp
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from natsort import natsorted

# ----------------------- Global Variable Initialization -----------------------
global_file_path_lv = ""
global_file_path_cv = ""
input_file_label_cv = None
file_label_lv = None
filename_step4 = ""
df_step4 = None
rapid_scan_var = None


# GUI Buttons (initialized later)
combine_csv_button = None
sort_button = None
time_resolved_csv_button = None
rename_columns_cv_button = None
rename_columns_lv_button = None
rename_time_button = None
process_background_data_button = None

# These variables will be set in the settings panels
global_t_eq_cv = None
global_e_begin_cv = None
global_e_vertex1_cv = None
global_e_vertex2_cv = None
global_scan_rate_cv = None
global_num_scans_cv = None

global_t_eq_lv = None
global_e_begin_lv = None
global_e_end_lv = None
global_scan_rate_lv = None


# ------------------------------------------------------------------ helpers
def _popup_if_network_error(exc, parent=None, filelabel="file"):
    """
    Return True iff *exc* looks like WinError 59 (“unexpected network error”)
    and shows a custom, high-visibility popup.  Otherwise returns False so
    the caller can fall back to the normal error box.
    """
    msg_lower = str(exc).lower()
    win_err = getattr(exc, "winerror", None)

    is_network = (win_err == 59) or ("network error" in msg_lower)
    if not is_network:
        return False  # let caller handle as usual

    # ------------------------ bespoke modal dialog -------------------------
    root = parent or window  # main window fallback
    dlg = tk.Toplevel(root)
    dlg.title("Network drive detected")
    dlg.transient(root)  # stay on top
    dlg.grab_set()  # modal
    dlg.resizable(False, False)

    PAD = 12
    body = tk.Frame(dlg, padx=PAD, pady=PAD)
    body.pack()

    # Big heading
    tk.Label(
        body, text="Unable to read the " + filelabel,
        font=("Helvetica", 20, "bold"), fg="red"
    ).pack(anchor="w", pady=(0, PAD))

    # Yellow highlighted advice
    tk.Label(
        body,
        text=("Please copy the data to a LOCAL disk\n"
              "or a OneDrive-synced folder and run again."),
        font=("Helvetica", 18, "bold"),
        bg="yellow", justify="left", wraplength=560
    ).pack(anchor="w", pady=(0, PAD))

    # Original Windows message (smaller, grey)
    tk.Label(
        body, text=f"Windows reported:\n{exc}",
        font=("Helvetica", 12), fg="gray40", justify="left", wraplength=560
    ).pack(anchor="w")

    # OK button
    btn = ttk.Button(dlg, text="OK", command=dlg.destroy)
    btn.pack(pady=(PAD, 0))
    btn.focus_set()

    # Centre over parent
    dlg.update_idletasks()
    w, h = dlg.winfo_width(), dlg.winfo_height()
    px = root.winfo_rootx() + (root.winfo_width() - w) // 2
    py = root.winfo_rooty() + (root.winfo_height() - h) // 2
    dlg.geometry(f"{w}x{h}+{px}+{py}")

    root.wait_window(dlg)  # block until closed
    return True  # tell caller we handled it


# ----------------------- Helper Function for CV Voltage Calculation -----------------------
def calculate_cv_voltage(t, params):
    E_begin = params['E_begin']
    E_vertex1 = params['E_vertex1']
    E_vertex2 = params['E_vertex2']
    sr = params['scan_rate']
    T_eq = params['T_eq']

    if t < T_eq:
        return E_begin

    if E_begin == E_vertex2:
        t1 = abs(E_vertex1 - E_begin) / sr
        T_cycle = 2 * t1
        t_in_cycle = (t - T_eq) % T_cycle
        if t_in_cycle < t1:
            return E_begin + (E_vertex1 - E_begin) * (t_in_cycle / t1)
        else:
            return E_vertex1 + (E_begin - E_vertex1) * ((t_in_cycle - t1) / t1)
    else:
        t1 = abs(E_vertex1 - E_begin) / sr
        t2 = abs(E_vertex2 - E_vertex1) / sr
        t3 = abs(E_vertex1 - E_begin) / sr
        T_cycle = t1 + t2 + t3
        t_in_cycle = (t - T_eq) % T_cycle
        if t_in_cycle < t1:
            return E_begin + (E_vertex1 - E_begin) * (t_in_cycle / t1)
        elif t_in_cycle < t1 + t2:
            return E_vertex1 + (E_vertex2 - E_vertex1) * ((t_in_cycle - t1) / t2)
        else:
            return E_vertex2 + (E_begin - E_vertex2) * ((t_in_cycle - t1 - t2) / t3)


# ----------------------- Helper function to check header in a CSV file -----------------------
def file_has_header(file_path, header_keywords=["wavenumbers", "cm^-1", "absorbance", "a.u."]):
    try:
        with open(file_path, "r") as f:
            first_line = f.readline().lower()
        return any(keyword in first_line for keyword in header_keywords)
    except Exception:
        return False


# ----------------------- STEP 1 : Convert selected SPA files -----------------------
def convert_spa_folder_individual():
    # ► pick one or many *.spa files rather than a folder
    spa_paths = filedialog.askopenfilenames(
        title="Select SPA file(s) to convert",
        filetypes=[("Thermo OMNIC SPA files", "*.spa")],
        parent=window
    )
    if not spa_paths:  # user hit Cancel
        return

    # progress window ---------------------------------------------------------
    pwin = tk.Toplevel(window)
    pwin.title("Converting SPA → CSV")
    ttk.Label(pwin, text="Converting each SPA to its own CSV…").pack(pady=(10, 0))
    pb = ttk.Progressbar(pwin, orient="horizontal", length=400, mode="determinate")
    pb["maximum"] = len(spa_paths)
    pb.pack(padx=20, pady=10)
    pwin.update()

    # worker ------------------------------------------------------------------
    def _task(path):
        ds = scp.read_omnic(path)
        if ds is None:
            raise ValueError("Unsupported .spa file")
        out_csv = os.path.splitext(path)[0] + ".csv"
        spec = ds if ds.ndim == 1 else ds[0]
        spec.write_csv(out_csv)

    errors, any_error, network_error_seen = [], False, False

    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_task, p): os.path.basename(p) for p in spa_paths}
        done = 0
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as e:
                any_error = True
                handled = False
                if not network_error_seen:
                    handled = _popup_if_network_error(e, parent=window, filelabel="SPA file")
                    if handled:
                        network_error_seen = True
                if not handled:
                    errors.append(f"{name}: {e}")
            done += 1
            pb["value"] = done
            pwin.update_idletasks()

    pwin.destroy()

    if any_error:
        if errors:
            messagebox.showerror("Errors while converting SPA files",
                                 "\n".join(errors), parent=window)
        # else only WinError-59 pop-up already shown once
    else:
        messagebox.showinfo("Done", "All selected SPA files converted successfully.",
                            parent=window)


# ----------------------- Convert OMNIC SRS Files -----------------------
def convert_srs_file_combined():
    file_path = filedialog.askopenfilename(
        title="Select an SRS File",
        filetypes=[("SRS files", "*.srs")],
        parent=window
    )
    if not file_path:
        return

    try:
        ds = scp.read_omnic(file_path)
        if ds is None or ds.ndim != 2:
            raise ValueError("Expected a 2D dataset from SRS.")

        # only flip if NOT rapid scan
        if not rapid_scan_var.get():
            ds.data = ds.data[:, ::-1]

        x = ds.x.data
        n_spec, n_pts = ds.data.shape
        base, _ = os.path.splitext(file_path)
        out_csv = f"{base}_combined.csv"

        pwin = tk.Toplevel(window)
        pwin.title("Exporting SRS → CSV")
        ttk.Label(pwin, text="Writing rows to CSV…").pack(pady=(10, 0))
        pb = ttk.Progressbar(pwin, orient="horizontal", length=400, mode="determinate")
        pb["maximum"] = n_pts
        pb.pack(padx=20, pady=10)
        pwin.update()

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Wavenumber"] + [f"Spectrum {i + 1}" for i in range(n_spec)])
            for i in range(n_pts):
                writer.writerow([x[i]] + ds.data[:, i].tolist())
                pb["value"] = i + 1
                pwin.update_idletasks()

        pwin.destroy()
        messagebox.showinfo(
            "Success",
            f"{n_spec} spectra × {n_pts} points written to:\n{out_csv}",
            parent=window
        )
    except Exception as e:
        if not _popup_if_network_error(e, parent=window, filelabel="SRS file"):
            messagebox.showerror("Error", str(e), parent=window)


# ----------------------- Step 2 Functions: CSV Combination -----------------------
def combine_csv_files_lazy(folder_path: str) -> pl.LazyFrame:
    csv_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the selected folder.")
    combined = None
    for fn in csv_files:
        path = os.path.join(folder_path, fn)
        skip = 1 if file_has_header(path) else 0
        lf = (pl.scan_csv(path, has_header=False, skip_rows=skip, new_columns=["Wavenumber", fn])
              .with_columns((pl.col("Wavenumber") // 0.1 * 0.1).alias("Wavenumber")))
        combined = lf if combined is None else combined.join(lf, how="full", on="Wavenumber")
    return combined


def combine_series_csv_to_csv():
    folder = filedialog.askdirectory(title="Select Folder with CSV Files")
    if not folder: return

    status_label.config(text="Processing Series CSV...", fg="blue")
    combine_csv_button.config(state=tk.DISABLED)
    window.update_idletasks()

    try:
        combined_lazy = combine_csv_files_lazy(folder)
        out_default = os.path.join(folder, "combined_series.csv")
        save_path = filedialog.asksaveasfilename(
            initialfile=out_default, defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], title="Save combined CSV", parent=window
        )
        if save_path:
            combined_lazy.sink_csv(save_path)
            messagebox.showinfo("Success", f"Data saved as {save_path}.", parent=window)
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=window)
    finally:
        status_label.config(text="Completed", fg="green")
        combine_csv_button.config(state=tk.NORMAL)


def sort_spectral_columns():
    file_path = filedialog.askopenfilename(
        title="Select Combined CSV File to Sort",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path: return

    status_label.config(text="Sorting Spectral Columns...", fg="blue")
    window.update_idletasks()
    sort_button.config(state=tk.DISABLED)
    if file_has_header(file_path):
        df = pd.read_csv(file_path, header=None, skiprows=1)
    else:
        df = pd.read_csv(file_path, header=None)
    wavenumber_col = df.pop(0)
    sorted_df = df.reindex(sorted(df.columns), axis=1)
    sorted_df.insert(0, "Wavenumber", wavenumber_col)
    sorted_file_path = os.path.splitext(file_path)[0] + "_sorted.csv"
    sorted_df.to_csv(sorted_file_path, index=False, header=False)
    messagebox.showinfo("Success", f"Sorted data saved as {sorted_file_path}.", parent=window)
    status_label.config(text="Completed", fg="green")
    sort_button.config(state=tk.NORMAL)


# ----------------------- Step 2b: Combine Time-Resolved CSV -----------------------
def extract_time_value(filename):
    match = re.search(r't\s*=\s*([\d.]+)', filename)
    return float(match.group(1)) if match else None


def combine_time_resolved_csv_to_csv():
    global time_resolved_csv_button
    folder_path = filedialog.askdirectory(title="Select Folder with Time-Resolved CSV Files")
    if not folder_path: return

    status_label.config(text="Processing Time-Resolved CSV...", fg="blue")
    time_resolved_csv_button.config(state=tk.DISABLED)
    window.update_idletasks()

    csv_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith('.csv') and "static" not in f.lower()]
    if not csv_files:
        messagebox.showerror("Error", "No suitable CSV files found.", parent=window)
        return

    combined_df = None
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        time_val = extract_time_value(file)
        if time_val is None:
            messagebox.showwarning("Skipping File", f"Could not extract time from: {file}", parent=window)
            continue
        skiprows = 1 if file_has_header(file_path) else 0
        df = pd.read_csv(file_path, header=None, skiprows=skiprows)
        if df.shape[1] < 2:
            continue
        df.columns = ['Wavenumber', f"{time_val:.2f}"]
        df['Wavenumber'] = (df['Wavenumber'] // 0.1) * 0.1
        combined_df = df if combined_df is None else pd.merge(combined_df, df, on='Wavenumber', how='outer')

    if combined_df is None:
        messagebox.showerror("Error", "No valid data to combine.", parent=window)
    else:
        time_cols = sorted([c for c in combined_df.columns if c != 'Wavenumber'],
                           key=lambda x: float(x))
        combined_df = combined_df[['Wavenumber'] + time_cols]
        save_as_csv_pandas(combined_df, folder_path)

    status_label.config(text="Completed", fg="green")
    time_resolved_csv_button.config(state=tk.NORMAL)


def save_as_csv_pandas(combined_data, folder_path, default_name_base="combined"):
    default_file = os.path.join(folder_path, f"{default_name_base}.csv")
    save_path = filedialog.asksaveasfilename(
        initialfile=default_file, defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")], title="Save data"
    )
    if save_path:
        combined_data.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Data saved as {save_path}.", parent=window)


# ----------------------- Step 3 Functions: Rename Columns -----------------------
def get_cv_settings():
    global t_eq_entry_cv, e_begin_entry_cv, e_vertex1_entry_cv, e_vertex2_entry_cv, scan_rate_entry_cv, num_scans_entry_cv, global_file_path_cv, input_file_label_cv

    def select_input_file():
        global global_file_path_cv
        selected = filedialog.askopenfilename(
            title="Select CV Input File", filetypes=[("CSV files", "*.csv")]
        )
        if selected:
            global_file_path_cv = selected
            input_file_label_cv.config(text=os.path.basename(selected))
            _update_cv_potential_change()

    def _update_cv_potential_change():
        try:
            t_eq = float(t_eq_entry_cv.get())
            eb = float(e_begin_entry_cv.get())
            v1 = float(e_vertex1_entry_cv.get())
            v2 = float(e_vertex2_entry_cv.get())
            sr = float(scan_rate_entry_cv.get())
            ns = int(num_scans_entry_cv.get())
            if not global_file_path_cv:
                return
            if not (min(v1, v2) <= eb <= max(v1, v2)):
                return
            df = pd.read_csv(global_file_path_cv)
            n_spec = len(df.columns) - 1
            if eb == v2:
                cycle_range = 2 * abs(v1 - eb)
            else:
                cycle_range = abs(v1 - eb) + abs(v2 - v1) + abs(eb - v2)
            total_time = t_eq + ns * (cycle_range / sr)
            delta_t = total_time / n_spec
            change = sr * delta_t
            potential_change_label_cv.config(text=f"Potential Change per Spectrum: {change:.6f} V/sec")
            globals().update({
                'global_t_eq_cv': t_eq,
                'global_e_begin_cv': eb,
                'global_e_vertex1_cv': v1,
                'global_e_vertex2_cv': v2,
                'global_scan_rate_cv': sr,
                'global_num_scans_cv': ns
            })
        except:
            pass

    # Header
    tk.Label(settings_frame_cv,
             text="Enter CV Settings", fg="blue", font=("Helvetica", 10, "bold")
             ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    # External trigger note immediately under header
    tk.Label(settings_frame_cv,
             text="If using external trigger, T_eq = 0.",
             fg="purple", font=("Helvetica", 10, "bold")
             ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))

    # Field labels and entries shifted down by 1 row
    labels = ["T equilibrium (s):", "E begin (V):", "E Vertex1 (V):", "E Vertex2 (V):",
              "Scan rate (V/s):", "Number of scans:"]
    for idx, text in enumerate(labels, start=2):
        tk.Label(settings_frame_cv, text=text, font=("Helvetica", 10, "bold")
                 ).grid(row=idx, column=0, sticky="w", pady=2)

    t_eq_entry_cv = tk.Entry(settings_frame_cv)
    e_begin_entry_cv = tk.Entry(settings_frame_cv)
    e_vertex1_entry_cv = tk.Entry(settings_frame_cv)
    e_vertex2_entry_cv = tk.Entry(settings_frame_cv)
    scan_rate_entry_cv = tk.Entry(settings_frame_cv)
    num_scans_entry_cv = tk.Entry(settings_frame_cv)
    entries = [t_eq_entry_cv, e_begin_entry_cv, e_vertex1_entry_cv,
               e_vertex2_entry_cv, scan_rate_entry_cv, num_scans_entry_cv]
    for idx, entry in enumerate(entries, start=2):
        entry.grid(row=idx, column=1, pady=2)
        entry.bind('<FocusOut>', lambda e: _update_cv_potential_change())

    # File selector stays at row 8
    tk.Button(settings_frame_cv,
              text="Select CV Input File",
              command=select_input_file,
              bg="lightgray"
              ).grid(row=8, column=0, columnspan=2, pady=4)
    input_file_label_cv = tk.Label(settings_frame_cv,
                                   text="No file selected", fg="black")
    input_file_label_cv.grid(row=9, column=0, columnspan=2, pady=2)


def rename_columns_cv():
    global global_file_path_cv
    if not global_file_path_cv:
        messagebox.showerror("Error", "No CV file selected", parent=window)
        return

    status_label.config(text="Renaming CV Headers...", fg="blue")
    rename_columns_cv_button.config(state=tk.DISABLED)
    window.update_idletasks()

    try:
        df = pd.read_csv(global_file_path_cv)
        n_spec = len(df.columns) - 1
        params = {
            "E_begin": global_e_begin_cv,
            "E_vertex1": global_e_vertex1_cv,
            "E_vertex2": global_e_vertex2_cv,
            "scan_rate": global_scan_rate_cv,
            "T_eq": global_t_eq_cv
        }
        if global_e_begin_cv == global_e_vertex2_cv:
            t1 = abs(global_e_vertex1_cv - global_e_begin_cv) / global_scan_rate_cv
            cycle = 2 * t1
        else:
            t1 = abs(global_e_vertex1_cv - global_e_begin_cv) / global_scan_rate_cv
            t2 = abs(global_e_vertex2_cv - global_e_vertex1_cv) / global_scan_rate_cv
            t3 = t1
            cycle = t1 + t2 + t3

        total = global_t_eq_cv + global_num_scans_cv * cycle
        delta = total / n_spec

        new_cols = ["Wavenumber"]
        for i in range(n_spec):
            t = i * delta
            v = (global_e_begin_cv if t < global_t_eq_cv
                 else calculate_cv_voltage(t, params))
            new_cols.append(f"{v:.4f} V")

        df.columns = new_cols
        out = os.path.splitext(global_file_path_cv)[0] + "_full_renamed_cv.csv"
        df.to_csv(out, index=False)
        messagebox.showinfo("Success", f"Saved: {out}", parent=window)
        status_label.config(text="Completed", fg="green")
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=window)
        status_label.config(text="Error", fg="red")
    finally:
        rename_columns_cv_button.config(state=tk.NORMAL)


def get_lv_settings():
    global t_eq_entry_lv, e_begin_entry_lv, e_end_entry_lv, scan_rate_entry_lv, global_file_path_lv, file_label_lv

    def select_input_file_lv():
        global global_file_path_lv
        selected = filedialog.askopenfilename(
            title="Select LV Input File", filetypes=[("CSV files", "*.csv")]
        )
        if selected:
            global_file_path_lv = selected
            file_label_lv.config(text=os.path.basename(selected))
            _update_lv_potential_change()

    def _update_lv_potential_change():
        try:
            t_eq = float(t_eq_entry_lv.get())
            eb = float(e_begin_entry_lv.get())
            ee = float(e_end_entry_lv.get())
            sr = float(scan_rate_entry_lv.get())
            if not global_file_path_lv or eb == ee:
                return
            df = pd.read_csv(global_file_path_lv)
            n_spec = len(df.columns) - 1
            ramp_time = abs(ee - eb) / sr
            total_time = t_eq + ramp_time
            delta_t = total_time / n_spec
            change = sr * delta_t
            potential_change_label_lv.config(text=f"Potential Change per Spectrum: {change:.6f} V/sec")
            globals().update({
                'global_t_eq_lv': t_eq,
                'global_e_begin_lv': eb,
                'global_e_end_lv': ee,
                'global_scan_rate_lv': sr
            })
        except:
            pass

    # Header
    tk.Label(settings_frame_lv,
             text="Enter LV Settings", fg="blue", font=("Helvetica", 10, "bold")
             ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

    # External trigger note at row 1
    tk.Label(settings_frame_lv,
             text="If using external trigger, T_eq = 0.",
             fg="purple", font=("Helvetica", 10, "bold")
             ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))

    # Field labels and entries start at row 2
    labels = ["T equilibrium (s):", "E begin (V):", "E end (V):", "Scan rate (V/s):"]
    for idx, text in enumerate(labels, start=2):
        tk.Label(settings_frame_lv, text=text, font=("Helvetica", 10, "bold")
                 ).grid(row=idx, column=0, sticky="w", pady=2)

    t_eq_entry_lv = tk.Entry(settings_frame_lv)
    e_begin_entry_lv = tk.Entry(settings_frame_lv)
    e_end_entry_lv = tk.Entry(settings_frame_lv)
    scan_rate_entry_lv = tk.Entry(settings_frame_lv)
    entries = [t_eq_entry_lv, e_begin_entry_lv, e_end_entry_lv, scan_rate_entry_lv]
    for idx, entry in enumerate(entries, start=2):
        entry.grid(row=idx, column=1, pady=2)
        entry.bind('<FocusOut>', lambda e: _update_lv_potential_change())

    # File selector at row 6
    tk.Button(settings_frame_lv,
              text="Select LV Input File",
              command=select_input_file_lv,
              bg="lightgray"
              ).grid(row=6, column=0, columnspan=2, pady=4)
    file_label_lv = tk.Label(settings_frame_lv, text="No file selected", fg="black")
    file_label_lv.grid(row=7, column=0, columnspan=2, pady=2)


def rename_columns_lv():
    global global_file_path_lv
    if not global_file_path_lv:
        messagebox.showerror("Error", "No LV file selected", parent=window)
        return

    status_label.config(text="Renaming LV Headers...", fg="blue")
    rename_columns_lv_button.config(state=tk.DISABLED)
    window.update_idletasks()

    try:
        df = pd.read_csv(global_file_path_lv)
        n_spec = len(df.columns) - 1
        ramp = abs(global_e_end_lv - global_e_begin_lv) / global_scan_rate_lv
        total = global_t_eq_lv + ramp
        delta = total / n_spec

        new_cols = ["Wavenumber"]
        for i in range(n_spec):
            t = i * delta
            if t < global_t_eq_lv:
                v = global_e_begin_lv
            else:
                tr = t - global_t_eq_lv
                if global_e_end_lv > global_e_begin_lv:
                    v = min(global_e_end_lv, global_e_begin_lv + global_scan_rate_lv * tr)
                else:
                    v = max(global_e_end_lv, global_e_begin_lv - global_scan_rate_lv * tr)
            new_cols.append(f"{v:.4f} V")

        df.columns = new_cols
        out = os.path.splitext(global_file_path_lv)[0] + "_renamed_lv.csv"
        df.to_csv(out, index=False)
        messagebox.showinfo("Success", f"Saved: {out}", parent=window)
        status_label.config(text="Completed", fg="green")
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=window)
        status_label.config(text="Error", fg="red")
    finally:
        rename_columns_lv_button.config(state=tk.NORMAL)


# ----------------------- Step 3C : Time-based header renaming -----------------------
def rename_headers_based_on_time():
    fn = filedialog.askopenfilename(
        title="Select CSV for Time Rename",
        filetypes=[("CSV files", "*.csv")],
        parent=window,
    )
    if not fn:
        return

    status_label.config(text="Renaming Time-based Headers…", fg="blue")
    rename_time_button.config(state=tk.DISABLED)
    window.update_idletasks()

    try:
        # ── load data ─────────────────────────────────────────────────────────
        df = pd.read_csv(fn)

        # ── ask the user for total duration ──────────────────────────────────
        total = simpledialog.askfloat(
            "Input",
            "Total Time Collected (seconds):",
            parent=window,
        )
        if total is None or total <= 0:
            raise ValueError("Total time must be a positive number.")

        # ── compute Δt and mid-points ────────────────────────────────────────
        ncols = len(df.columns) - 1  # spectra only (exclude wavenumber)
        dt = total / ncols  # acquisition window per spectrum

        # Mid-point labels: (i + 0.5)·Δt  →  0.5·Δt, 1.5·Δt, …
        new_headers = ["Wavenumber"] + [f"{(i + 0.5) * dt:.3f}s" for i in range(ncols)]
        df.columns = new_headers

        # ── save ----------------------------------------------------------------
        out = os.path.splitext(fn)[0] + "_renamed_Time.csv"
        df.to_csv(out, index=False)

        messagebox.showinfo("Success", f"Saved: {out}", parent=window)
        status_label.config(text="Completed", fg="green")

    except Exception as e:
        messagebox.showerror("Error", str(e), parent=window)
        status_label.config(text="Error", fg="red")

    finally:
        rename_time_button.config(state=tk.NORMAL)


# ────────────────────────────────
# Polars helper for fast time-header normalization
def normalize_time_headers_polars(csv_path: str, ref_time: float) -> str:
    df = pl.read_csv(csv_path)
    old = df.columns
    new = [old[0]] + [
        (lambda m: f"{float(m.group(1)) - ref_time:.3f}s")(re.search(r"([-+]?\d*\.?\d+)\s*s?$", c))
        if re.search(r"([-+]?\d*\.?\d+)\s*s?$", c) else c
        for c in old[1:]
    ]
    df = df.rename({o: n for o, n in zip(old, new)})
    out = os.path.splitext(csv_path)[0] + f"_normRef{ref_time:.3f}.csv"
    df.write_csv(out)
    return out


def run_normalize():
    path = norm_file_path.get()
    if not path or not os.path.isfile(path):
        messagebox.showerror("Error", "Select a valid CSV", parent=window)
        return
    try:
        ref = float(norm_ref_entry.get())
    except:
        messagebox.showerror("Error", "Enter numeric reference time", parent=window)
        return
    try:
        out = normalize_time_headers_polars(path, ref)
        messagebox.showinfo("Done", f"Saved:\n{out}", parent=window)
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=window)


# ----------------------- Step 4: Background Reprocessing -----------------------
def bg_processing():
    file_path = filedialog.askopenfilename(title="Select Input File",
                                           filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    try:
        status_label.config(text="Reprocessing Background...", fg="blue")
        process_background_data_button.config(state=tk.DISABLED)
        window.update_idletasks()
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            messagebox.showerror("Error", "Unsupported file format.", parent=window)
            raise ValueError("Unsupported file format")
        column_window = tk.Tk()
        column_window.title("Select Column")
        max_column_name_length = max(len(str(col)) for col in df.columns)
        combobox_width = max(30, max_column_name_length // 2)
        column_label = ttk.Label(column_window, text="Choose a column:")
        column_label.pack(pady=5)
        sanitized_column_names = [str(col) for col in df.columns if col != "Wavenumber"]
        column_combobox = ttk.Combobox(column_window, values=sanitized_column_names, width=combobox_width)
        column_combobox.pack(pady=5)
        column_combobox.current(0)  # default to the first spectral column

        def on_confirm():
            chosen_column = column_combobox.get()
            if not chosen_column:
                messagebox.showerror("Error", "No column selected.", parent=window)
                return
            column_window.destroy()
            process_and_save(chosen_column, file_path, df)
            status_label.config(text="Completed", fg="green")
            process_background_data_button.config(state=tk.NORMAL)

        confirm_button = ttk.Button(column_window, text="Confirm", command=on_confirm)
        confirm_button.pack(pady=5)
        column_window.geometry(f"{combobox_width * 10}x150")
        column_window.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}", parent=window)
        status_label.config(text="Idle", fg="green")
    finally:
        process_background_data_button.config(state=tk.NORMAL)


def process_and_save(chosen_column, file_path, df):
    # subtract in one go:
    #   take all spectral cols, subtract the background column along rows
    spectral = df.drop(columns="Wavenumber")
    background = df[chosen_column]
    subtracted = spectral.subtract(background, axis=0)

    # re-attach the Wavenumber column up front
    processed = pd.concat(
        [df[["Wavenumber"]], subtracted],
        axis=1,
        copy=False  # avoid an extra copy if you like
    )

    out = os.path.splitext(file_path)[0] + f"_{chosen_column}.csv"
    processed.to_csv(out, index=False)
    messagebox.showinfo("Success", f"File saved as {out}", parent=window)
    status_label.config(text="Idle", fg="green")


# ----------------------- STEP 5: Skip Spectral Columns (Linear & CV-Aware) -----------------------
# ——————————— Linear Skip Functionality ———————————
def linear_skip_run():
    path = lin_file_path.get()
    if not path or not os.path.isfile(path):
        messagebox.showerror("Error", "Select a valid CSV", parent=root)
        return
    try:
        n = int(lin_n_entry.get())
        if n < 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Enter non-negative integer for n", parent=root)
        return

    df = pd.read_csv(path)
    cols = df.shape[1]
    if cols < 3:
        messagebox.showerror("Error", "CSV needs ≥3 columns", parent=root)
        return

    keep = [0, 1]
    cycle = n + 1
    for i in range(2, cols - 1):
        if (i - 2) % cycle == n:
            keep.append(i)
    keep.append(cols - 1)

    out = df.iloc[:, keep]
    base = os.path.splitext(path)[0]
    out_path = f"{base}_skip{n}.csv"
    out.to_csv(out_path, index=False)
    messagebox.showinfo("Done", f"Saved: {out_path}", parent=root)


def linear_browse():
    path = filedialog.askopenfilename(title="Select CSV for Linear Skip",
                                      filetypes=[("CSV", "*.csv")])
    if path:
        lin_file_path.set(path)


# ————————— CV-Aware Skip Functionality —————————
def cv_browse_and_suggest():
    path = filedialog.askopenfilename(title="Select CV-renamed CSV",
                                      filetypes=[("CSV", "*.csv")])
    if not path:
        return
    cv_file_path.set(path)
    # parse headers for voltage suggestions
    with open(path, newline='') as f:
        hdr = next(csv.reader(f))
    volts = []
    for h in hdr[1:]:
        m = re.search(r"[-+]?\d*\.\d+|\d+", h)
        if m:
            volts.append(float(m.group()))
    if len(volts) < 2:
        return
    dV_sugg = (max(volts) - min(volts)) / (len(volts) - 1)
    dv_entry.delete(0, tk.END)
    dv_entry.insert(0, f"{dV_sugg:.4f}")
    sugg_dv_lbl.config(text=f"Suggested: {dV_sugg:.4f} V")
    pos_entry.delete(0, tk.END)
    pos_entry.insert(0, f"{dV_sugg:.4f}")
    sugg_pos_lbl.config(text=f"±{dV_sugg:.4f}")
    neg_entry.delete(0, tk.END)
    neg_entry.insert(0, f"{dV_sugg:.4f}")
    sugg_neg_lbl.config(text=f"±{dV_sugg:.4f}")


def cv_skip_run():
    path = cv_file_path.get()
    if not path or not os.path.isfile(path):
        messagebox.showerror("Error", "Select a valid CSV", parent=root)
        return

    # collect inputs
    try:
        Eb = float(e_begin_entry.get())
        Ev1 = float(e_v1_entry.get())
        Ev2 = float(e_v2_entry.get())
        sr = float(sr_entry.get())
        T_eq = float(teq_entry.get())
        nsc = int(nsc_entry.get())
        dV = float(dv_entry.get())
        tol_p = float(pos_entry.get())
        tol_n = float(neg_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Enter valid numeric CV & ΔV/tolerance", parent=root)
        return

    # read headers
    with open(path, newline='') as f:
        hdr = next(csv.reader(f))
    voltages = []
    for h in hdr[1:]:
        m = re.search(r"[-+]?\d*\.\d+|\d+", h)
        if not m:
            messagebox.showerror("Error", f"Bad header '{h}'", parent=root)
            return
        voltages.append(float(m.group()))

    last = len(hdr) - 1
    keep = {0, 1, last}

    # ---------------------------------------------------------------- vertex windows
    for Ev in (Eb, Ev1, Ev2):
        cands = [i for i, v in enumerate(voltages, start=1)
                 if Ev - tol_n <= v <= Ev + tol_p]
        if cands:
            keep.add(cands[0])  # first match (leave as-is)
            keep.add(cands[-1])  # last match (leave as-is)

    # ---------------------------------------------------------------- uniform ΔV sampling
    vmin, vmax = min(voltages), max(voltages)
    steps = np.arange(vmin, vmax + dV / 2, dV)  # target grid

    for target in steps:
        # indices (1-based) whose header lies inside the tolerance window
        cands = [i for i, v in enumerate(voltages, start=1)
                 if target - tol_n <= v <= target + tol_p]
        if cands:
            keep.add(cands[0])  # ← ascending
            keep.add(cands[-1])  # ← descending

    keep_list = sorted(keep)

    # stream write out
    base, _ = os.path.splitext(path)
    out_path = f"{base}_dV{dV:.3f}.csv"
    with open(path, newline='') as fin, open(out_path, 'w', newline='') as fout:
        rdr = csv.reader(fin)
        wtr = csv.writer(fout)
        orig = next(rdr)
        wtr.writerow([orig[i] for i in keep_list])
        for row in rdr:
            wtr.writerow([row[i] for i in keep_list])

    messagebox.showinfo("Done", f"Saved: {out_path}", parent=root)


# ───────────────────────────────────────────────────  Time-Crop helpers
crop_values = []              # list of numeric header values

def crop_browse():
    global crop_values
    path = filedialog.askopenfilename(
        title="Select CSV to crop",
        filetypes=[("CSV files", "*.csv")],
        parent=window
    )
    if not path:
        return
    crop_file_path.set(path)

    # parse the first header row, extract all numbers
    with open(path, newline="") as fh:
        hdr = next(csv.reader(fh))
    vals = []
    for h in hdr[1:]:
        m = re.search(r"([-+]?\d*\.?\d+)", h)
        if m:
            vals.append(float(m.group(1)))
    if not vals:
        messagebox.showwarning("No numeric headers",
                               "Couldn't find any numbers in the headers.",
                               parent=window)
        crop_values = []
        return

    # sort & initialize
    crop_values = sorted(vals)
    n = len(crop_values)
    crop_start.set(crop_values[0])
    crop_stop.set(crop_values[-1])
    sug_min_lbl.config(text=f"Min: {crop_values[0]:.3f}")
    sug_max_lbl.config(text=f"Max: {crop_values[-1]:.3f}")
    range_slider.configure(range=(0, n-1))
    range_slider.set_values(0, n-1)

def _snap_idx(val):
    return min(range(len(crop_values)),
               key=lambda i: abs(crop_values[i] - val))

def on_crop_entry(event=None):
    try:
        i0 = _snap_idx(crop_start.get())
        i1 = _snap_idx(crop_stop.get())
        if i1 < i0:
            i0, i1 = i1, i0
        range_slider.set_values(i0, i1)
        crop_start.set(crop_values[i0])
        crop_stop.set(crop_values[i1])
    except Exception:
        pass

def crop_run():
    path = crop_file_path.get()
    if not path or not os.path.isfile(path):
        messagebox.showerror("Error", "Select a valid CSV", parent=window)
        return
    if not crop_values:
        messagebox.showwarning("No data loaded", "Please browse a file first.", parent=window)
        return

    i0, i1 = range_slider.get_values()
    keep = [0] + list(range(1 + i0, 1 + i1 + 1))
    keep.sort()

    base, _ = os.path.splitext(path)
    out = f"{base}_{crop_values[i0]:.3f}-{crop_values[i1]:.3f}.csv"
    with open(path, newline="") as fin, open(out, "w", newline="") as fout:
        rdr, wtr = csv.reader(fin), csv.writer(fout)
        hdr = next(rdr)
        wtr.writerow([hdr[k] for k in keep])
        for row in rdr:
            wtr.writerow([row[k] for k in keep])

    messagebox.showinfo("Done", f"Saved:\n{out}", parent=window)

# ----------------------- Main GUI: Notebook Tabbed Layout -----------------------
root = tk.Tk()
root.title("FTIR Data Processing_V11")
root.geometry("1100x750")
window = root
rapid_scan_var = tk.BooleanVar(value=False)
crop_file_path   = tk.StringVar()
crop_start       = tk.DoubleVar()
crop_stop        = tk.DoubleVar()

def exit_application() -> None:
    root.quit()


# ------------------------------------------------------------------ styling
style = ttk.Style(root)
style.theme_use("default")
style.configure("TNotebook.Tab", padding=[10, 5], font=("Helvetica", 10, "bold"))
style.map("TNotebook.Tab", background=[("selected", "gray")])

# ------------------------------------------------------------- header / status
header_frame = tk.Frame(root, padx=20, pady=20)
header_frame.pack(fill="x")

tk.Label(
    header_frame, text="FTIR Data Processing_V11",
    font=("Helvetica", 12, "bold"), fg="blue"
).pack(pady=(0, 10))

status_label = tk.Label(
    header_frame, text="Idle",
    font=("Helvetica", 12, "bold"), fg="green"
)
status_label.pack(pady=(0, 10))

# ------------------------------------------------------------- top-level tabs
main_notebook = ttk.Notebook(root)
main_notebook.pack(fill="both", expand=True, padx=10, pady=10)

# =================================================================== STEP 1 ===
step1_tab = tk.Frame(main_notebook)
main_notebook.add(step1_tab, text="Step 1 : Convert SPA/SRS")

tk.Label(step1_tab, text="SPA & SRS to CSV Converter",
         font=("Helvetica", 14, "bold"), fg="blue").pack(pady=10)

tk.Label(
    step1_tab,
    text=("Important: Save data files on local disk or OneDrive; "
          "network drives will not work."),
    font=("Helvetica", 14, "bold"), fg="red", bg="yellow",
    wraplength=500
).pack(pady=10)

# ───────── SPA Conversion ─────────
spa_frame = tk.LabelFrame(step1_tab, text="Convert SPA Files", padx=10, pady=10)
spa_frame.pack(padx=20, pady=(0,10))   # no fill, centered by default

tk.Button(spa_frame,
          text="Convert SPA files → each CSV",
          font=("Helvetica", 11),
          bg="sky blue",
          command=convert_spa_folder_individual
).pack(pady=5)                        # centered

# ───────── SRS Conversion ─────────
srs_frame = tk.LabelFrame(step1_tab, text="Convert SRS Files", padx=10, pady=10)
srs_frame.pack(padx=20, pady=(0,20))  # no fill, centered by default

tk.Button(srs_frame,
          text="Convert SRS → combined CSV",
          font=("Helvetica", 11),
          bg="sky blue",
          command=convert_srs_file_combined
).pack(pady=5)                        # centered

tk.Checkbutton(srs_frame,
               text="Rapid Scan/Ifg reprocessed (Flip wavenumbers)",
               variable=rapid_scan_var,
               font=("Helvetica", 12, "bold"),
               padx=10, pady=6
).pack(pady=5)                        # centered

tk.Label(step1_tab,
         text="Converted CSV files will be saved in the same directory as the source file.",
         font=("Helvetica", 11, "italic"), fg="blue").pack(pady=10)

# =================================================================== STEP 2 ===
step2_tab = tk.Frame(main_notebook)
main_notebook.add(step2_tab, text="Step 2 : Combine CSV Files")

tk.Label(step2_tab, text="Combine CSV files",
         font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))

step2_nb = ttk.Notebook(step2_tab)  # ← use this name consistently
step2_nb.pack(fill="both", expand=True, padx=10, pady=10)

# ---- 2A : Series collection
series_tab = tk.Frame(step2_nb)
step2_nb.add(series_tab, text="Series Collection CSV")

tk.Label(series_tab, text="Combine Series Collection CSV Files",
         font=("Helvetica", 11, "bold"), fg="blue").pack(pady=10)

combine_csv_button = tk.Button(series_tab, text="Click to Combine CSV Files",
                               command=combine_series_csv_to_csv, bg="sky blue")
combine_csv_button.pack(pady=5)

tk.Label(series_tab,
         text="Note : Sort if >5k files; headers may not align.",
         font=("Helvetica", 11, "italic"), fg="blue").pack(pady=5)

sort_button = tk.Button(series_tab, text="Sort Spectral Columns",
                        command=sort_spectral_columns, bg="sky blue")
sort_button.pack(pady=5)

# ---- 2B : Time-resolved (step-scan)
time_tab = tk.Frame(step2_nb)
step2_nb.add(time_tab, text="Step-Scan Time-Resolved CSV")

tk.Label(time_tab, text="Combine Step-Scan Time-Resolved CSV Files",
         font=("Helvetica", 11, "bold"), fg="blue").pack(pady=10)

time_resolved_csv_button = tk.Button(
    time_tab, text="Click to Combine SSTR CSV Files",
    command=combine_time_resolved_csv_to_csv, bg="sky blue"
)
time_resolved_csv_button.pack(pady=5)

# =================================================================== STEP 3 ===
step3_tab = tk.Frame(main_notebook)
main_notebook.add(step3_tab, text="Step 3 : Rename Columns")

tk.Label(step3_tab, text="Rename Column Headers",
         font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))

step3_nb = ttk.Notebook(step3_tab)
step3_nb.pack(fill="both", expand=True, padx=10, pady=10)

# ------------------- 3A : CV
cv_tab = tk.Frame(step3_nb)
step3_nb.add(cv_tab, text="CV Voltage Range")

settings_frame_cv = tk.Frame(cv_tab, padx=10, pady=10)
settings_frame_cv.pack(fill="x", anchor="nw")

get_cv_settings()

potential_change_label_cv = tk.Label(
    cv_tab, bg="green yellow", fg="black",
    text="Potential Change per Spectrum: 0.000000 V/sec",
    font=("Helvetica", 12)
)
potential_change_label_cv.pack(pady=10)

rename_columns_cv_button = tk.Button(
    cv_tab, text="Rename CV Column Headers",
    bg="sky blue", command=rename_columns_cv
)
rename_columns_cv_button.pack(pady=5)

# ------------------- 3B : LV
lv_tab = tk.Frame(step3_nb)
step3_nb.add(lv_tab, text="LV Voltage Range")

settings_frame_lv = tk.Frame(lv_tab, padx=10, pady=10)
settings_frame_lv.pack(fill="x", anchor="nw")

get_lv_settings()

potential_change_label_lv = tk.Label(
    lv_tab, bg="green yellow", fg="black",
    text="Potential Change per Spectrum: 0.000000 V/sec",
    font=("Helvetica", 12)
)
potential_change_label_lv.pack(pady=10)

rename_columns_lv_button = tk.Button(
    lv_tab, text="Rename LV Column Headers",
    bg="sky blue", command=rename_columns_lv
)
rename_columns_lv_button.pack(pady=5)

# ------------------- 3 C : Time-based
tb_tab = tk.Frame(step3_nb)
step3_nb.add(tb_tab, text="Time-Based Labeling")

tk.Label(
    tb_tab,
    text=("Use this if you know the total scan duration\n"
          "and the tool will rename each spectral column"
          "to the MID-POINT of its acquisition time.\n"
          "(e.g. 0.125 s, 0.375 s, 0.625 s …)"),
    font=("Helvetica", 11, "italic", "bold"),
    fg="blue",
    wraplength=700  # ← line added
).pack(pady=15)

rename_time_button = tk.Button(
    tb_tab, text="Rename Headers Based on Time Intervals",
    bg="sky blue", command=rename_headers_based_on_time
)
rename_time_button.pack(pady=15)
# — new normalize-headers subsection —
norm_frame = tk.LabelFrame(tb_tab, text="Normalize Time Headers", padx=10, pady=10)
norm_frame.pack(fill="x", padx=20, pady=10)

tk.Label(
    norm_frame,
    text="Use this if you want to subtract a reference time from each spectral header\n"
         "(e.g. to shift your t₀ → 0 and rename 0.500s → 0.000s, etc.)",
    font=("Helvetica", 10, "italic", "bold"),
    fg="blue",
    justify="left",
    wraplength=500
).grid(row=0, column=0, columnspan=3, pady=(0, 8))

norm_file_path = tk.StringVar()
tk.Label(norm_frame, text="CSV File:", font=("Arial", 10)).grid(row=1, column=0, sticky="e")
tk.Entry(norm_frame, textvariable=norm_file_path, width=40).grid(row=1, column=1, padx=5)
tk.Button(norm_frame, text="Browse…", bg="lightgray",
          command=lambda: norm_file_path.set(
              filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV", "*.csv")])
          )
          ).grid(row=1, column=2)

tk.Label(norm_frame, text="Reference t₀ (s):", font=("Arial", 10)).grid(row=2, column=0, sticky="e", pady=5)
norm_ref_entry = tk.Entry(norm_frame, width=12)
norm_ref_entry.grid(row=2, column=1, sticky="w")

tk.Button(norm_frame, text="Normalize Headers", bg="sky blue", command=run_normalize) \
    .grid(row=3, column=1, pady=10)
# =================================================================== STEP 4 ===
step4_tab = tk.Frame(main_notebook)
main_notebook.add(step4_tab, text="Step 4 : Reprocess Background")

tk.Label(step4_tab, text="Reprocess Background",
         font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))

tk.Label(step4_tab,
         text=("Logic : Choose a background column and subtract it from all "
               "others\n(I_reprocessed = I_original − I_background)"),
         font=("Helvetica", 11), fg="blue",
         wraplength=500, justify="center").pack(pady=(0, 10))

process_background_data_button = tk.Button(
    step4_tab, text="Reprocess Background",
    bg="sky blue", command=bg_processing
)
process_background_data_button.pack(pady=10)

# =================================================================== STEP 5 ===
step5_tab = tk.Frame(main_notebook)
main_notebook.add(step5_tab, text="Step 5 : Skip/Crop Spectral Columns")

tk.Label(step5_tab, text="Skip/Crop Spectral Columns",
         font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))

step5_nb = ttk.Notebook(step5_tab)
step5_nb.pack(fill="both", expand=True, padx=10, pady=10)

# -------------------------- 5A : Linear Skip ---------------------------------
lin_tab = tk.Frame(step5_nb)
step5_nb.add(lin_tab, text="Linear Skip")

# Make the whole tab center-friendly
lin_tab.columnconfigure(0, weight=1)

# --- centred instruction banner ---------------------------------------------
tk.Label(
    lin_tab,
    text=("Linear skip keeps the first and last spectral columns, then skips "
          "every n columns and keeps the (n + 1)-th.\n\n"
          "Example: n = 2  →  keeps columns 1, 4, 7, 10 … plus the last."),
    font=("Arial", 12),
    fg="blue",
    wraplength=600,
    justify="center"
).grid(row=0, column=0, pady=(0, 15), sticky="n")

# --- form area ---------------------------------------------------------------
form = tk.Frame(lin_tab)  # inner frame, auto-centred by parent grid
form.grid(row=1, column=0)

lin_file_path = tk.StringVar()

tk.Label(form, text="CSV File:", font=("Arial", 10, "bold")
         ).grid(row=0, column=0, sticky="e", padx=5, pady=5)
tk.Entry(form, textvariable=lin_file_path, width=50
         ).grid(row=0, column=1, padx=5)
tk.Button(form, text="Browse…", command=linear_browse
          ).grid(row=0, column=2, padx=5)

tk.Label(form, text="Number of columns to skip:", font=("Arial", 10, "bold")
         ).grid(row=1, column=0, sticky="e", padx=5, pady=5)
lin_n_entry = tk.Entry(form, width=10)
lin_n_entry.grid(row=1, column=1, sticky="w")

tk.Button(
    form, text="Run Linear Skip", bg="sky blue",
    command=linear_skip_run
).grid(row=2, column=1, pady=20)

# -------------------------- 5 B : CV-Aware Skip
cv_tab5 = tk.Frame(step5_nb)
step5_nb.add(cv_tab5, text="CV-Aware Skip")

cv_file_path = tk.StringVar()
# --- 5B : CV-Aware Skip (SIDE HELP PANEL) ------------------------------
cv_help = tk.Text(
    cv_tab5, font=("Arial", 12), fg="blue", width=38, height=16, wrap="word",
    bg=root.cget("bg"), relief="flat", borderwidth=0
)
cv_help.insert(
    "1.0",
    "CV-aware skip keeps:\n"
    " • The first and last spectra\n"
    " • Both spectra at each defined vertex (E_begin, E_vertex1, E_vertex2)\n"
    " • The spectrum within every ΔV interval "
    "across the cycle (using the ± tolerances)\n\n"
    "Result: a compact data set that still covers all key voltages "
    "and samples uniformly in between."
)
cv_help.configure(state="disabled")
# Place to the right of the parameter grid
cv_help.grid(row=0, column=3, rowspan=20, sticky="nw", padx=(20, 0), pady=5)

tk.Label(cv_tab5, text="CSV File:", font=("Arial", 10, "bold")
         ).grid(row=0, column=0, sticky="e", padx=5, pady=5)
tk.Entry(cv_tab5, textvariable=cv_file_path, width=50
         ).grid(row=0, column=1, padx=5)
tk.Button(cv_tab5, text="Browse…", command=cv_browse_and_suggest
          ).grid(row=0, column=2, padx=5)

# ---------------------------------------------------------------- CV parameters
labels_cv = ["T_eq (s)", "E_begin (V)", "E_vertex1 (V)", "E_vertex2 (V)",
             "Scan rate (V/s)", "Number of Scans"]
entries_cv = {}

for r, lbl in enumerate(labels_cv, start=1):
    tk.Label(cv_tab5, text=lbl + ":", font=("Arial", 10, "bold")
             ).grid(row=r, column=0, sticky="e", padx=5, pady=3)
    ent = tk.Entry(cv_tab5, width=12)
    ent.grid(row=r, column=1, sticky="w")
    entries_cv[lbl] = ent

# expose the entries to cv_skip_run()
teq_entry = entries_cv["T_eq (s)"]
e_begin_entry = entries_cv["E_begin (V)"]
e_v1_entry = entries_cv["E_vertex1 (V)"]
e_v2_entry = entries_cv["E_vertex2 (V)"]
sr_entry = entries_cv["Scan rate (V/s)"]
nsc_entry = entries_cv["Number of Scans"]

# ------------- ΔV +tolerance widgets
row = len(labels_cv) + 1

tk.Label(cv_tab5, text="ΔV Interval (V):", font=("Arial", 10, "bold")
         ).grid(row=row, column=0, sticky="e", padx=5, pady=5)
dv_entry = tk.Entry(cv_tab5, width=12)
dv_entry.grid(row=row, column=1, sticky="w")
sugg_dv_lbl = tk.Label(cv_tab5, text="Suggested: —", fg="gray")
sugg_dv_lbl.grid(row=row, column=2, sticky="w")

row += 1
tk.Label(cv_tab5, text="+ve Tol (V):", font=("Arial", 10, "bold")
         ).grid(row=row, column=0, sticky="e", padx=5, pady=3)
pos_entry = tk.Entry(cv_tab5, width=12)
pos_entry.grid(row=row, column=1, sticky="w")
sugg_pos_lbl = tk.Label(cv_tab5, text="± —", fg="gray")
sugg_pos_lbl.grid(row=row, column=2, sticky="w")

row += 1
tk.Label(cv_tab5, text="-ve Tol (V):", font=("Arial", 10, "bold")
         ).grid(row=row, column=0, sticky="e", padx=5, pady=3)
neg_entry = tk.Entry(cv_tab5, width=12)
neg_entry.grid(row=row, column=1, sticky="w")
sugg_neg_lbl = tk.Label(cv_tab5, text="± —", fg="gray")
sugg_neg_lbl.grid(row=row, column=2, sticky="w")

# ----------------- Run button
tk.Button(cv_tab5, text="Run CV-Aware Skip", bg="sky blue",
          command=cv_skip_run).grid(row=row + 1, column=1, pady=20)


# ----------------- End of CV aware skip -----------------
# ───────────────────────────────────────────────────  Crop tab UI
class RangeSlider(tk.Canvas):
    def __init__(self, parent, length=400, handle_radius=8, **kwargs):
        super().__init__(parent, width=length, height=2*handle_radius+20, **kwargs)
        self.length = length
        self.hr = handle_radius
        self.low_idx = 0
        self.high_idx = 1
        self.min_idx = 0
        self.max_idx = 1

        # draw track + handles
        self.track = self.create_line(0, self.hr+10, length, self.hr+10, width=4)
        self.handle_low  = self.create_oval(0,0,0,0, fill="gray")
        self.handle_high = self.create_oval(0,0,0,0, fill="gray")

        # mouse events
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>",    self._on_drag)
        self.active = None

    def configure(self, **kwargs):
        if 'range' in kwargs:
            self.min_idx, self.max_idx = kwargs.pop('range')
        super().configure(**kwargs)

    def set_values(self, i0, i1):
        # clamp to [min_idx,max_idx]
        self.low_idx  = max(self.min_idx, min(i0, self.max_idx))
        self.high_idx = max(self.min_idx, min(i1, self.max_idx))
        self._draw_handles()

    def get_values(self):
        return (self.low_idx, self.high_idx)

    def _idx_to_x(self, idx):
        span = self.max_idx - self.min_idx
        if span <= 0: return 0
        return ((idx - self.min_idx) / span) * self.length

    def _x_to_idx(self, x):
        span = self.max_idx - self.min_idx
        frac = min(max(x, 0), self.length) / self.length
        return int(round(self.min_idx + frac*span))

    def _draw_handles(self):
        for idx, handle in ((self.low_idx, self.handle_low),
                            (self.high_idx, self.handle_high)):
            x = self._idx_to_x(idx)
            self.coords(handle, x-self.hr, 10, x+self.hr, 10+2*self.hr)

    def _on_press(self, ev):
        # pick the nearer handle
        lx = self._idx_to_x(self.low_idx)
        hx = self._idx_to_x(self.high_idx)
        self.active = 'low' if abs(ev.x-lx) < abs(ev.x-hx) else 'high'

    def _on_drag(self, ev):
        idx = self._x_to_idx(ev.x)
        if self.active == 'low':
            idx = min(idx, self.high_idx)
            self.low_idx = max(self.min_idx, idx)
        else:
            idx = max(idx, self.low_idx)
            self.high_idx = min(self.max_idx, idx)
        self._draw_handles()

        # if you want to sync entry widgets:
        crop_start.set(crop_values[self.low_idx])
        crop_stop.set(crop_values[self.high_idx])

crop_tab = tk.Frame(step5_nb)
step5_nb.add(crop_tab, text="Crop Data")

# Row 0: file selector
tk.Label(crop_tab, text="CSV File:", font=("Arial", 10, "bold"))\
  .grid(row=0, column=0, sticky="e", padx=5, pady=5)
tk.Entry(crop_tab, textvariable=crop_file_path, width=40)\
  .grid(row=0, column=1, padx=5, pady=5)
tk.Button(crop_tab, text="Browse…", command=crop_browse)\
  .grid(row=0, column=2, padx=5, pady=5)

# Row 1–2: suggestions
sug_min_lbl = tk.Label(crop_tab, text="Min: —", fg="gray")
sug_min_lbl.grid(row=1, column=2, sticky="w")
sug_max_lbl = tk.Label(crop_tab, text="Max: —", fg="gray")
sug_max_lbl.grid(row=2, column=2, sticky="w")

# Row 1: start entry
tk.Label(crop_tab, text="Start:", font=("Arial", 10, "bold"))\
  .grid(row=1, column=0, sticky="e", padx=5)
start_ent = tk.Entry(crop_tab, textvariable=crop_start, width=10)
start_ent.grid(row=1, column=1, sticky="w")
start_ent.bind("<Return>", on_crop_entry)

# Row 2: stop entry
tk.Label(crop_tab, text="Stop:", font=("Arial", 10, "bold"))\
  .grid(row=2, column=0, sticky="e", padx=5)
stop_ent = tk.Entry(crop_tab, textvariable=crop_stop, width=10)
stop_ent.grid(row=2, column=1, sticky="w")
stop_ent.bind("<Return>", on_crop_entry)

# Row 3: range slider
range_slider = RangeSlider(crop_tab)
range_slider.grid(row=3, column=0, columnspan=3, pady=10)

# Row 4: run button
tk.Button(crop_tab, text="Run Crop", bg="sky blue", command=crop_run)\
  .grid(row=4, column=1, pady=10)

# Help text
tk.Label(crop_tab,
    text="Keeps Wavenumber + all spectra columns whose header "
         "value lies between Start and Stop (snapped to nearest header).",
    font=("Arial", 12), fg="blue", wraplength=400, justify="left"
).grid(row=1, column=3, rowspan=3, padx=20)
# -------------------------------------------------------------- footer / exit
tk.Button(root, text="Exit Application", command=exit_application,
          bg="tomato").pack(pady=10)

tk.Label(root,
         text="Made by Pavithra Gunasekaran + ChatGPT "
              "(pavijovi3@gmail.com)",
         font=("Helvetica", 9)).pack(pady=10)

# -------------------------------------------------------------- main loop
root.mainloop()
