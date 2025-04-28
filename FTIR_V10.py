import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import spectrochempy as scp
import pandas as pd
import polars as pl
import numpy as np
from natsort import natsorted

# ----------------------- Global Variable Initialization -----------------------
global_file_path_lv = ""
global_file_path_cv = ""
filename_step4 = ""
df_step4 = None

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
global_potential_change_per_spectrum_cv = None

global_t_eq_lv = None
global_e_begin_lv = None
global_e_end_lv = None
global_scan_rate_lv = None
global_potential_change_per_spectrum_lv = None


# ----------------------- Helper Function for CV Voltage Calculation -----------------------
def calculate_cv_voltage(t, params):
    """
    Calculate potential at an absolute time t (in seconds) based on CV parameters.
    If t is within the equilibrium period, returns E_begin.
    For t >= Teq, the time is mapped into a cycle and linear interpolation is applied.
    Two cases are considered:
       - Case 1 (E_begin == E_vertex2): two-stage cycle.
       - Case 2 (E_begin != E_vertex2): three-stage cycle.
    """
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
    """
    Checks if the first line of the file contains any of the header keywords.
    Returns True if so, False otherwise.
    """
    try:
        with open(file_path, "r") as f:
            first_line = f.readline().lower()
        return any(keyword in first_line for keyword in header_keywords)
    except Exception:
        # If there is any problem reading the file, assume no header.
        return False


# ----------------------- STEP 1: Convert OMNIC Files -----------------------
def convert_spa_folder():
    """
    Converts each .spa file in a selected folder to CSV.
    Each file is assumed to be 1D (a single spectrum).
    If a header row is present (as from a manual conversion), it is skipped.
    """
    folder_path = filedialog.askdirectory(title="Select Folder Containing SPA Files")
    if not folder_path:
        return

    spa_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.spa')]
    if not spa_files:
        messagebox.showerror("Error", "No .spa files found in the selected folder.")
        return

    errors = []
    for filename in spa_files:
        full_path = os.path.join(folder_path, filename)
        try:
            dataset = scp.read_omnic(full_path)
            if dataset is None:
                raise ValueError("SpectroChemPy returned None. File may not be supported.")
            base, _ = os.path.splitext(full_path)
            out_csv = base + ".csv"
            if dataset.ndim == 1:
                dataset.write_csv(out_csv)
            else:
                # If multiple spectra exist, save the first spectrum.
                dataset[0].write_csv(out_csv)
            print(f"Converted: {filename}")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    if errors:
        messagebox.showerror("Conversion Errors", "\n".join(errors))
    else:
        messagebox.showinfo("Conversion Successful", "All SPA files converted successfully.")


def convert_srs_file():
    """
    Converts a .srs file into one combined CSV file with spectra as separate columns.
    Corrects absorbance inversion by reversing the dataset data.
    Displays a progress bar during processing.
    """
    file_path = filedialog.askopenfilename(
        title="Select an SRS File", filetypes=[("SRS files", "*.srs")]
    )
    if not file_path:
        return
    try:
        dataset = scp.read_omnic(file_path)
        if dataset is None or dataset.ndim != 2:
            raise ValueError("Expected a 2D dataset from SRS file.")

        # Correct absorbance: reverse along the second axis
        dataset.data = dataset.data[:, ::-1]

        x = dataset.x.data            # Wavenumber axis (after correction)
        base, _ = os.path.splitext(file_path)
        output_csv = f"{base}_combined.csv"

        # Setup progress bar window
        progress_win = tk.Toplevel()
        progress_win.title("Exporting Spectra...")
        tk.Label(progress_win, text="Exporting spectra to combined CSV...").pack(pady=10)
        progress = ttk.Progressbar(progress_win, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = dataset.shape[0]
        progress["value"] = 0
        progress_win.update()

        # Stack all spectra correctly
        y_data = []  # Collect y data separately
        for i in range(dataset.shape[0]):
            subds = dataset[i].squeeze()  # Make sure we squeeze dimensions properly
            y = subds.data                # Now y is shape (n_points,)
            y_data.append(y)
            progress["value"] += 1
            progress_win.update()

        # Now combine x and all y spectra
        combined_array = np.column_stack([x] + y_data)
        headers = ["Wavenumber"] + [f"Spectrum {i+1}" for i in range(dataset.shape[0])]
        np.savetxt(output_csv, combined_array, delimiter=",", header=",".join(headers), comments='')

        progress_win.destroy()
        messagebox.showinfo("Conversion Successful", f"{dataset.shape[0]} spectra saved to:\n{output_csv}")



# ----------------------- Step 2 Functions: CSV Combination -----------------------
def combine_csv_files(folder_path):
    """
    Combines CSV files in a folder using Polars.
    For each file, if the first line is detected as a header (contains keywords such as
    "wavenumbers", "cm^-1", "absorbance", "a.u."), then that row is skipped.
    The files are joined on the "Wavenumber" column.
    """
    csv_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        return "No CSV files found in the selected folder."
    combined_data = None
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        print(f"Processing file {i + 1}/{len(csv_files)}: {csv_file}")
        # Check if this file includes a header row.
        skip_rows = 1 if file_has_header(file_path) else 0
        try:
            # Read the file with Polars.
            # We force has_header=False and supply new_columns so that data are read as numerical data.
            df = pl.read_csv(file_path, has_header=False, skip_rows=skip_rows, new_columns=["Wavenumber", csv_file])
            # Optionally, truncate wavenumbers to 1 decimal place.
            df = df.with_columns([(pl.col("Wavenumber") // 0.1 * 0.1).alias("Wavenumber")])
        except Exception as e:
            return f"Error reading {csv_file}: {str(e)}"
        if combined_data is None:
            combined_data = df
        else:
            # Rename the Wavenumber column in df temporarily to avoid duplicate names.
            df = df.rename({"Wavenumber": "Wavenumber_temp"})
            combined_data = combined_data.join(df, left_on="Wavenumber", right_on="Wavenumber_temp", how="full")
            combined_data = combined_data.drop("Wavenumber_temp")
    return combined_data


def save_as_csv_polars(combined_data, folder_path, default_name_base="combined"):
    default_file_name = os.path.join(folder_path, default_name_base + ".csv")
    save_path = filedialog.asksaveasfilename(initialfile=default_file_name,
                                             defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             title="Save data", parent=window)
    if save_path:
        combined_data.write_csv(save_path)
        messagebox.showinfo("Success", f"Data saved as {save_path}.", parent=window)


def combine_series_csv_to_csv():
    """
    This function combines CSV files (from series conversion) using Polars.
    It uses combine_csv_files() and then saves the combined DataFrame as CSV.
    """
    folder_path = filedialog.askdirectory(title="Select Folder with CSV Files")
    if folder_path:
        status_label.config(text="Processing Series CSV...", fg="blue")
        window.update_idletasks()
        combine_csv_button.config(state=tk.DISABLED)
        window.update_idletasks()
        combined_data = combine_csv_files(folder_path)
        if isinstance(combined_data, str):
            messagebox.showerror("Error", combined_data, parent=window)
        else:
            save_as_csv_polars(combined_data, folder_path)
        status_label.config(text="Completed", fg="green")
        combine_csv_button.config(state=tk.NORMAL)


def sort_spectral_columns():
    global sort_button
    """
    Sorts the columns of a combined CSV file.
    If the file contains header text in the first row, it is removed.
    """
    file_path = filedialog.askopenfilename(title="Select Combined CSV File to Sort",
                                           filetypes=[("CSV files", "*.csv")])
    if file_path:
        status_label.config(text="Sorting Spectral Columns...", fg="blue")
        window.update_idletasks()
        sort_button.config(state=tk.DISABLED)
        window.update_idletasks()
        # Check for header text in the first row.
        if file_has_header(file_path):
            df = pd.read_csv(file_path, header=None, skiprows=1)
        else:
            df = pd.read_csv(file_path, header=None)
        # Assume first column holds the wavenumbers.
        wavenumber_col = df.pop(0)
        sorted_df = df.reindex(sorted(df.columns), axis=1)
        sorted_df.insert(0, "Wavenumber", wavenumber_col)
        sorted_file_path = os.path.splitext(file_path)[0] + "_sorted.csv"
        sorted_df.to_csv(sorted_file_path, index=False, header=False)
        messagebox.showinfo("Success", f"Sorted data saved as {sorted_file_path}.", parent=window)
        status_label.config(text="Completed", fg="green")
        window.update_idletasks()
        sort_button.config(state=tk.NORMAL)


# ----------------------- Step 2: b) Combine Time-Resolved CSV Files -----------------------
def extract_time_value(filename):
    """
    Extract the numeric time value from the filename, e.g., 't = 0.00' from 'file_t = 0.00.csv'.
    """
    match = re.search(r't\s*=\s*([\d.]+)', filename)
    if match:
        return float(match.group(1))  # Returns float for sorting purposes
    return None


def combine_time_resolved_csv_to_csv():
    global time_resolved_csv_button
    folder_path = filedialog.askdirectory(title="Select Folder with Time-Resolved CSV Files")
    if not folder_path:
        return

    status_label.config(text="Processing Time-Resolved CSV...", fg="blue")
    time_resolved_csv_button.config(state=tk.DISABLED)
    window.update_idletasks()

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv') and "static" not in f.lower()]
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

        # Read with or without header
        skiprows = 1 if file_has_header(file_path) else 0
        df = pd.read_csv(file_path, header=None, skiprows=skiprows)

        if df.shape[1] < 2:
            continue

        df.columns = ['Wavenumber', f"{time_val:.2f}"]

        # Truncate wavenumber to 1 decimal place to help match up
        df['Wavenumber'] = (df['Wavenumber'] // 0.1) * 0.1

        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Wavenumber', how='outer')

    if combined_df is None:
        messagebox.showerror("Error", "No valid data to combine.", parent=window)
        return

    # Sort columns by time (excluding Wavenumber)
    time_cols = [col for col in combined_df.columns if col != 'Wavenumber']
    time_cols_sorted = sorted(time_cols, key=lambda x: float(x.replace("s", "")))
    combined_df = combined_df[['Wavenumber'] + time_cols_sorted]

    save_as_csv_pandas(combined_df, folder_path)
    status_label.config(text="Completed", fg="green")
    time_resolved_csv_button.config(state=tk.NORMAL)


def save_as_csv_pandas(combined_data, folder_path, default_name_base="combined"):
    default_file_name = os.path.join(folder_path, default_name_base + ".csv")
    save_path = filedialog.asksaveasfilename(initialfile=default_file_name,
                                             defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             title="Save data")
    if save_path:
        combined_data.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Data saved as {save_path}.", parent=window)


# ----------------------- Step 3 Functions: Rename Columns (CV, LV, and Time) -----------------------
def get_cv_settings():
    global t_eq_entry_cv, e_begin_entry_cv, e_vertex1_entry_cv, e_vertex2_entry_cv, scan_rate_entry_cv, num_scans_entry_cv, global_file_path_cv

    def select_input_file():
        selected_path = filedialog.askopenfilename(title="Select CV Input File", filetypes=[("CSV files", "*.csv")])
        if selected_path:
            global global_file_path_cv
            global_file_path_cv = selected_path
            input_file_label_cv.config(text=os.path.basename(selected_path))

    def save_cv_settings():
        try:
            global global_t_eq_cv, global_e_begin_cv, global_e_vertex1_cv, global_e_vertex2_cv, global_scan_rate_cv, global_num_scans_cv
            global_t_eq_cv = float(t_eq_entry_cv.get())
            global_e_begin_cv = float(e_begin_entry_cv.get())
            global_e_vertex1_cv = float(e_vertex1_entry_cv.get())
            global_e_vertex2_cv = float(e_vertex2_entry_cv.get())
            global_scan_rate_cv = float(scan_rate_entry_cv.get())
            global_num_scans_cv = int(num_scans_entry_cv.get())
            if not global_file_path_cv:
                messagebox.showerror("Input Error", "Please select an input file.", parent=window)
                return
            if not (min(global_e_vertex1_cv, global_e_vertex2_cv) <= global_e_begin_cv <= max(global_e_vertex1_cv,
                                                                                              global_e_vertex2_cv)):
                messagebox.showerror("Input Error", "E_begin must be between E_vertex1 and E_vertex2", parent=window)
                return
            df = pd.read_csv(global_file_path_cv)
            num_spectra_total = len(df.columns) - 1
            if global_e_begin_cv == global_e_vertex2_cv:
                total_potential_range_cv = 2 * abs(global_e_vertex1_cv - global_e_begin_cv)
            else:
                total_potential_range_cv = abs(global_e_vertex1_cv - global_e_begin_cv) + abs(
                    global_e_vertex2_cv - global_e_vertex1_cv) + abs(global_e_begin_cv - global_e_vertex2_cv)
            total_time_per_cycle = total_potential_range_cv / global_scan_rate_cv
            T_total = global_t_eq_cv + (global_num_scans_cv * total_time_per_cycle)
            delta_t = T_total / num_spectra_total
            potential_change_label_cv.config(
                text=f"Potential Change per Spectrum: {(global_scan_rate_cv * delta_t):.6f} V/sec")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.", parent=window)

    tk.Label(settings_frame_cv,
             text="Enter CV Settings", fg="blue", font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                       sticky="w", pady=(0, 10))
    tk.Label(settings_frame_cv, text="T equilibrium (s):", font=("Helvetica", 10, "bold")).grid(row=1, column=0,
                                                                                                sticky="w", pady=2)
    tk.Label(settings_frame_cv, text="E begin (V):", font=("Helvetica", 10, "bold")).grid(row=2, column=0, sticky="w",
                                                                                          pady=2)
    tk.Label(settings_frame_cv, text="E Vertex1 (V):", font=("Helvetica", 10, "bold")).grid(row=3, column=0, sticky="w",
                                                                                            pady=2)
    tk.Label(settings_frame_cv, text="E Vertex2 (V):", font=("Helvetica", 10, "bold")).grid(row=4, column=0, sticky="w",
                                                                                            pady=2)
    tk.Label(settings_frame_cv, text="Scan rate (V/s):", font=("Helvetica", 10, "bold")).grid(row=5, column=0,
                                                                                              sticky="w", pady=2)
    tk.Label(settings_frame_cv, text="Number of scans:", font=("Helvetica", 10, "bold")).grid(row=6, column=0,
                                                                                              sticky="w", pady=2)
    t_eq_entry_cv = tk.Entry(settings_frame_cv)
    e_begin_entry_cv = tk.Entry(settings_frame_cv)
    e_vertex1_entry_cv = tk.Entry(settings_frame_cv)
    e_vertex2_entry_cv = tk.Entry(settings_frame_cv)
    scan_rate_entry_cv = tk.Entry(settings_frame_cv)
    num_scans_entry_cv = tk.Entry(settings_frame_cv)
    t_eq_entry_cv.grid(row=1, column=1, pady=2)
    e_begin_entry_cv.grid(row=2, column=1, pady=2)
    e_vertex1_entry_cv.grid(row=3, column=1, pady=2)
    e_vertex2_entry_cv.grid(row=4, column=1, pady=2)
    scan_rate_entry_cv.grid(row=5, column=1, pady=2)
    num_scans_entry_cv.grid(row=6, column=1, pady=2)
    tk.Button(settings_frame_cv, text='Select CV Input File', command=select_input_file, bg="lightgray").grid(row=7,
                                                                                                              column=0,
                                                                                                              pady=4)
    input_file_label_cv = tk.Label(settings_frame_cv, text="No file selected", fg="black")
    input_file_label_cv.grid(row=7, column=1, pady=4)
    tk.Button(settings_frame_cv, text='Save CV Settings', command=save_cv_settings, bg="green yellow").grid(row=8,
                                                                                                            column=1,
                                                                                                            pady=4)


def rename_columns_cv():
    global global_file_path_cv
    if global_file_path_cv:
        status_label.config(text="Renaming CV Headers...", fg="blue")
        rename_columns_cv_button.config(state=tk.DISABLED)
        window.update_idletasks()
        try:
            df = pd.read_csv(global_file_path_cv)
            num_spectra_total = len(df.columns) - 1
            params = {
                'E_begin': global_e_begin_cv,
                'E_vertex1': global_e_vertex1_cv,
                'E_vertex2': global_e_vertex2_cv,
                'scan_rate': global_scan_rate_cv,
                'T_eq': global_t_eq_cv
            }
            if global_e_begin_cv == global_e_vertex2_cv:
                t1 = abs(global_e_vertex1_cv - global_e_begin_cv) / global_scan_rate_cv
                T_cycle = 2 * t1
            else:
                t1 = abs(global_e_vertex1_cv - global_e_begin_cv) / global_scan_rate_cv
                t2 = abs(global_e_vertex2_cv - global_e_vertex1_cv) / global_scan_rate_cv
                t3 = abs(global_e_vertex1_cv - global_e_begin_cv) / global_scan_rate_cv
                T_cycle = t1 + t2 + t3
            T_total = global_t_eq_cv + (global_num_scans_cv * T_cycle)
            delta_t = T_total / num_spectra_total
            new_columns = ["Wavenumber"]
            for i in range(num_spectra_total):
                t = i * delta_t
                if t < global_t_eq_cv:
                    voltage = global_e_begin_cv
                else:
                    voltage = calculate_cv_voltage(t, params)
                new_columns.append(f"{voltage:.4f} V")
            df.columns = new_columns
            save_path_cv = os.path.splitext(global_file_path_cv)[0] + "_full_renamed_cv.csv"
            df.to_csv(save_path_cv, index=False)
            messagebox.showinfo("Success", f"CV headers renamed and file saved as {save_path_cv}.", parent=window)
            status_label.config(text="Completed", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=window)
            status_label.config(text="Error", fg="red")
        finally:
            rename_columns_cv_button.config(state=tk.NORMAL)


def get_lv_settings():
    global t_eq_entry_lv, e_begin_entry_lv, e_end_entry_lv, scan_rate_entry_lv, global_file_path_lv

    def select_input_file_lv():
        global global_file_path_lv
        global_file_path_lv = filedialog.askopenfilename(title="Select LV Input File",
                                                         filetypes=[("CSV files", "*.csv")])
        if global_file_path_lv:
            file_label_lv.config(text=os.path.basename(global_file_path_lv))

    def save_lv_settings():
        global global_t_eq_lv, global_e_begin_lv, global_e_end_lv, global_scan_rate_lv
        try:
            global_t_eq_lv = float(t_eq_entry_lv.get())
            global_e_begin_lv = float(e_begin_entry_lv.get())
            global_e_end_lv = float(e_end_entry_lv.get())
            global_scan_rate_lv = float(scan_rate_entry_lv.get())
            if global_e_begin_lv == global_e_end_lv:
                messagebox.showerror("Input Error", "E_begin must not equal E_end", parent=window)
                return
            if not global_file_path_lv:
                messagebox.showerror("Missing File", "Please select an input file.", parent=window)
                return
            df = pd.read_csv(global_file_path_lv)
            num_spectra_lv = len(df.columns) - 1
            ramp_time = abs(global_e_end_lv - global_e_begin_lv) / global_scan_rate_lv
            T_total = global_t_eq_lv + ramp_time
            delta_t = T_total / num_spectra_lv
            potential_change = global_scan_rate_lv * delta_t
            potential_change_label_lv.config(text=f"Potential Change per Spectrum: {potential_change:.6f} V/sec")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.", parent=window)

    tk.Label(settings_frame_lv,
             text="Enter LV Settings", fg="blue", font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                       sticky="w", pady=(0, 10))
    tk.Label(settings_frame_lv, text="T equilibrium (s):", font=("Helvetica", 10, "bold")).grid(row=1, column=0,
                                                                                                sticky="w", pady=2)
    tk.Label(settings_frame_lv, text="E begin (V):", font=("Helvetica", 10, "bold")).grid(row=2, column=0, sticky="w",
                                                                                          pady=2)
    tk.Label(settings_frame_lv, text="E end (V):", font=("Helvetica", 10, "bold")).grid(row=3, column=0, sticky="w",
                                                                                        pady=2)
    tk.Label(settings_frame_lv, text="Scan rate (V/s):", font=("Helvetica", 10, "bold")).grid(row=4, column=0,
                                                                                              sticky="w", pady=2)
    t_eq_entry_lv = tk.Entry(settings_frame_lv)
    e_begin_entry_lv = tk.Entry(settings_frame_lv)
    e_end_entry_lv = tk.Entry(settings_frame_lv)
    scan_rate_entry_lv = tk.Entry(settings_frame_lv)
    t_eq_entry_lv.grid(row=1, column=1, pady=2)
    e_begin_entry_lv.grid(row=2, column=1, pady=2)
    e_end_entry_lv.grid(row=3, column=1, pady=2)
    scan_rate_entry_lv.grid(row=4, column=1, pady=2)
    tk.Button(settings_frame_lv, text='Select LV Input File', command=select_input_file_lv, bg="lightgray").grid(row=5,
                                                                                                                 column=0,
                                                                                                                 pady=4)
    file_label_lv = tk.Label(settings_frame_lv, text="No file selected", fg="black")
    file_label_lv.grid(row=5, column=1, pady=4)
    tk.Button(settings_frame_lv, text='Save LV Settings', command=save_lv_settings, bg="green yellow").grid(row=6,
                                                                                                            column=1,
                                                                                                            pady=4)


def rename_columns_lv():
    global global_file_path_lv
    if global_file_path_lv:
        status_label.config(text="Renaming LV Headers...", fg="blue")
        rename_columns_lv_button.config(state=tk.DISABLED)
        window.update_idletasks()
        try:
            df = pd.read_csv(global_file_path_lv)
            num_spectra_lv = len(df.columns) - 1
            ramp_time = abs(global_e_end_lv - global_e_begin_lv) / global_scan_rate_lv
            T_total = global_t_eq_lv + ramp_time
            delta_t = T_total / num_spectra_lv
            new_columns_lv = ["Wavenumber"]
            for i in range(num_spectra_lv):
                t = i * delta_t
                if t < global_t_eq_lv:
                    voltage = global_e_begin_lv
                else:
                    t_ramp = t - global_t_eq_lv
                    if global_e_end_lv > global_e_begin_lv:
                        voltage = global_e_begin_lv + global_scan_rate_lv * t_ramp
                        if voltage > global_e_end_lv:
                            voltage = global_e_end_lv
                    else:
                        voltage = global_e_begin_lv - global_scan_rate_lv * t_ramp
                        if voltage < global_e_end_lv:
                            voltage = global_e_end_lv
                new_columns_lv.append(f"{voltage:.4f} V")
            df.columns = new_columns_lv
            save_path_lv = os.path.splitext(global_file_path_lv)[0] + "_renamed_lv.csv"
            df.to_csv(save_path_lv, index=False)
            messagebox.showinfo("Success", f"LV headers renamed and file saved as {save_path_lv}.", parent=window)
            status_label.config(text="Completed", fg="green")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=window)
            status_label.config(text="Error", fg="red")
        finally:
            rename_columns_lv_button.config(state=tk.NORMAL)


def rename_headers_based_on_time():
    global filename_step1
    filename_step1 = filedialog.askopenfilename(title="Select CSV File for Time Renaming",
                                                filetypes=[("CSV files", "*.csv")])
    if filename_step1:
        status_label.config(text="Renaming Time-based Headers...", fg="blue")
        rename_time_button.config(state=tk.DISABLED)
        window.update_idletasks()
        try:
            df_step1 = pd.read_csv(filename_step1)
            total_time = simpledialog.askfloat("Input", "Total Time Collected (seconds):")
            if total_time is None:
                return
            num_columns = len(df_step1.columns) - 1
            time_interval = total_time / num_columns
            new_headers = ['Wavenumber'] + [f"{i * time_interval:.2f}s" for i in range(num_columns)]
            df_step1.columns = new_headers
            renamed_filename = os.path.splitext(filename_step1)[0] + "_renamed.csv"
            df_step1.to_csv(renamed_filename, index=False)
            messagebox.showinfo("Success", f"Headers renamed and saved to {renamed_filename}.", parent=window)
            status_label.config(text="Completed", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Enter a valid number for total time.", parent=window)
            status_label.config(text="Error", fg="red")
        finally:
            rename_time_button.config(state=tk.NORMAL)


# ----------------------- Step 4 Function: Background Reprocessing -----------------------
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
    processed_sheet = pd.DataFrame()
    processed_sheet["Wavenumber"] = df["Wavenumber"]
    for column in df.columns[1:]:
        if column == chosen_column:
            processed_sheet[column] = 0
        else:
            processed_sheet[column] = df[column] - df[chosen_column]
    save_path = os.path.splitext(file_path)[0] + f"_{chosen_column}.csv"
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        f.write(','.join(df.columns) + '\n')
        processed_sheet.to_csv(f, index=False, header=False)
    messagebox.showinfo("Success", f"File saved as {save_path}", parent=window)
    status_label.config(text="Idle", fg="green")


# ----------------------- STEP 5: Reduce Spectral Columns -----------------------
def select_and_process_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    n = simpledialog.askinteger(
        "Input",
        "Enter how many spectra columns to skip between each kept column (n):",
        minvalue=0
    )
    if n is None:
        return

    try:
        df = pd.read_csv(file_path)
        total_cols = df.shape[1]
        if total_cols < 3:
            messagebox.showerror("Error", "CSV must have at least 3 columns.")
            return

        # Always keep first two columns
        keep = [0, 1]
        cycle = n + 1
        # For columns 2..(end-1), keep one every cycle
        for i in range(2, total_cols - 1):
            if (i - 2) % cycle == n:
                keep.append(i)
        # Always keep last column
        keep.append(total_cols - 1)

        df_reduced = df.iloc[:, keep]
        base, ext = os.path.splitext(os.path.basename(file_path))
        out = os.path.join(os.path.dirname(file_path), f"{base}_skip{n}.csv")
        df_reduced.to_csv(out, index=False)
        messagebox.showinfo("Success", f"Saved reduced file as:\n{os.path.basename(out)}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ----------------------- Main GUI: Notebook Tabbed Layout -----------------------
root = tk.Tk()
root.withdraw()


def exit_application():
    try:
        window.quit()
    except Exception as e:
        print(f"Error closing application: {str(e)}")


# ----------------------- Main GUI Setup -----------------------
root = tk.Tk()
root.withdraw()

window = tk.Tk()
window.title("FTIR Data Processing_V10")
window.geometry("650x800")

style = ttk.Style()
style.theme_use('default')
style.configure("TNotebook.Tab", padding=[10, 5], font=("Helvetica", 10, "bold"))
style.map("TNotebook.Tab", background=[("selected", "gray")])

header_frame = tk.Frame(window, padx=20, pady=20)
header_frame.pack(fill='x')
tk.Label(header_frame, text="FTIR Data Processing_V10", font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(0, 10))
status_label = tk.Label(header_frame, text="Idle", font=("Helvetica", 12, "bold"), fg="green")
status_label.pack(pady=(0, 10))

main_notebook = ttk.Notebook(window)
main_notebook.pack(fill='both', expand=True, padx=10, pady=10)

# STEP 1 Tab
step1_tab = tk.Frame(main_notebook)
main_notebook.add(step1_tab, text="Step 1: Convert SPA/SRS")
tk.Label(step1_tab, text="SPA & SRS to CSV Converter", font=("Helvetica", 14, "bold"), fg="blue").pack(pady=10)
tk.Label(step1_tab,
         text="Important — for this step, store both the program and its data on a "
              "local disk or OneDrive;\nnetwork drives will not work.",
         font=("Helvetica", 10, "bold"), fg="red").pack(pady=10)
tk.Label(step1_tab, text="Select one of the conversion options below:", font=("Helvetica", 12)).pack(pady=5)
convert_spa_button = tk.Button(step1_tab, text="Convert SPA Files (Folder)", font=("Helvetica", 10),
                               command=convert_spa_folder, bg="sky blue")
convert_spa_button.pack(pady=5)
convert_srs_button = tk.Button(step1_tab, text="Convert SRS File (Multi-spectrum)", font=("Helvetica", 10),
                               command=convert_srs_file, bg="sky blue")
convert_srs_button.pack(pady=5)
tk.Label(step1_tab, text="Converted CSV files will be saved in the same directory as the source file.",
         font=("Helvetica", 9, "italic"), fg="blue").pack(pady=10)

# STEP 2 Tab
step2_tab = tk.Frame(main_notebook)
main_notebook.add(step2_tab, text="Step 2: Combine CSV Files")
tk.Label(step2_tab, text="Combine CSV files", font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))
step2_notebook = ttk.Notebook(step2_tab)
step2_notebook.pack(fill='both', expand=True, padx=10, pady=10)

series_tab = tk.Frame(step2_notebook)
step2_notebook.add(series_tab, text="Series Collection CSV")
tk.Label(series_tab, text="Combine Series Collection CSV Files", font=("Helvetica", 10, "bold"), fg="blue").pack(
    pady=10)
combine_csv_button = tk.Button(series_tab, text="Click to Combine CSV Files", command=combine_series_csv_to_csv,
                               bg="sky blue")
combine_csv_button.pack(pady=5)
tk.Label(series_tab, text="Note: Sort if >5k files; headers may not align.", font=("Helvetica", 9, "italic"),
         fg="blue").pack(pady=5)
sort_button = tk.Button(series_tab, text="Sort Spectral Columns", command=sort_spectral_columns, bg="sky blue")
sort_button.pack(pady=5)

time_tab = tk.Frame(step2_notebook)
step2_notebook.add(time_tab, text="Step-Scan Time-Resolved CSV")
tk.Label(time_tab, text="Combine Step-Scan Time-Resolved CSV Files", font=("Helvetica", 10, "bold"), fg="blue").pack(
    pady=10)
time_resolved_csv_button = tk.Button(time_tab, text="Click to Combine SSTR CSV Files",
                                     command=combine_time_resolved_csv_to_csv, bg="sky blue")
time_resolved_csv_button.pack(pady=5)

# STEP 3 Tab
step3_tab = tk.Frame(main_notebook)
main_notebook.add(step3_tab, text="Step 3: Rename Columns")
tk.Label(step3_tab, text="Rename Column Headers", font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))
step3_notebook = ttk.Notebook(step3_tab)
step3_notebook.pack(fill='both', expand=True, padx=10, pady=10)

cv_tab = tk.Frame(step3_notebook)
step3_notebook.add(cv_tab, text="CV Voltage Range")
settings_frame_cv = tk.Frame(cv_tab, padx=10, pady=10)
settings_frame_cv.pack(fill='x', anchor='nw')
get_cv_settings()
potential_change_label_cv = tk.Label(cv_tab, text="Potential Change per Spectrum: 0.000000 V/sec", bg="white",
                                     fg="black", font=("Helvetica", 10))
potential_change_label_cv.pack(pady=10)
rename_columns_cv_button = tk.Button(cv_tab, text="Rename CV Column Headers", command=rename_columns_cv, bg="sky blue")
rename_columns_cv_button.pack(pady=5)

lv_tab = tk.Frame(step3_notebook)
step3_notebook.add(lv_tab, text="LV Voltage Range")
settings_frame_lv = tk.Frame(lv_tab, padx=10, pady=10)
settings_frame_lv.pack(fill='x', anchor='nw')
get_lv_settings()
potential_change_label_lv = tk.Label(lv_tab, text="Potential Change per Spectrum: 0.000000 V/sec", bg="white",
                                     fg="black", font=("Helvetica", 10))
potential_change_label_lv.pack(pady=10)
rename_columns_lv_button = tk.Button(lv_tab, text="Rename LV Column Headers", command=rename_columns_lv, bg="sky blue")
rename_columns_lv_button.pack(pady=5)

time_based_tab = tk.Frame(step3_notebook)
step3_notebook.add(time_based_tab, text="Time-Based Labeling")
tk.Label(time_based_tab, text="Use this if you know total scan duration\nto label headers like 0.00s, 0.25s...",
         font=("Helvetica", 9, "italic"), fg="purple").pack(pady=10)
rename_time_button = tk.Button(time_based_tab, text="Rename Headers Based on Time Intervals",
                               command=rename_headers_based_on_time, bg="sky blue")
rename_time_button.pack(pady=10)

# STEP 4 Tab
step4_tab = tk.Frame(main_notebook)
main_notebook.add(step4_tab, text="Reprocess Background")
tk.Label(step4_tab, text="Reprocess Background", font=("Helvetica", 12, "bold"), fg="blue").pack(pady=(10, 5))
tk.Label(step4_tab,
         text="Logic: Choose a background column and subtract it from all others\n(I_reprocessed = I_original - I_background)",
         font=("Helvetica", 10), fg="blue", wraplength=400, justify="center").pack(pady=(0, 10))
process_background_data_button = tk.Button(step4_tab, text="Reprocess Background", command=bg_processing, bg="sky blue")
process_background_data_button.pack(pady=10)

# Step 5 Tab: Reduce Spectral Columns
step5_tab = tk.Frame(main_notebook)
main_notebook.add(step5_tab, text="Skip Spectral Columns")
tk.Label(
    step5_tab,
    text="The output file keeps Wavenumber, first, last and every (n+1)th column",
    font=("Helvetica", 10, "bold"),
    fg="blue",
    wraplength=500,
    justify="center"
).pack(pady=(10, 5))
skip_button = tk.Button(
    step5_tab,
    text="Select the CSV file to reduce",
    width=25,
    height=2,
    command=select_and_process_file,
    bg="sky blue"
)
skip_button.pack(pady=10)
# Exit + Footer
tk.Button(window, text="Exit Application", command=exit_application, bg="tomato").pack(pady=10)
tk.Label(window, text="Made by Pavithra Gunasekaran + ChatGPT (pavijovi3@gmail.com)", font=("Helvetica", 8)).pack(
    pady=10)

window.mainloop()
