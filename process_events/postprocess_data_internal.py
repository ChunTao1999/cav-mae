import argparse
import os
import tkinter as tk
from tkinter import font
import glob
from PIL import Image, ImageTk
from natsort import natsorted
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import find_peaks
import pdb # for debug


class BoundingBoxApp:
    def __init__(self, root, data_path, event_type_path, prev_path, save_path, image_frame_size, frames_per_event):
        self.root = root
        self.data_path = data_path
        self.event_type_path = event_type_path
        self.prev_json_path = prev_path
        self.json_save_path = save_path
        self.get_rv_files_list()
        self.prepare_json()
        self.frame_size = image_frame_size
        self.frames_per_event = frames_per_event
        self.file_idx = 0
        self.canvas = None
        self.points = []
        self.event_label = ""
        self.bbox_list = []
        self.setup_ui()


    def get_rv_files_list(self):
         # get the session ids
        self.file_list = glob.glob(os.path.join(self.data_path, "undistorted_rv_annotated", "*.png"))
        self.file_list = natsorted(self.file_list)
        return 
    

    def prepare_json(self):
        shutil.copy(self.prev_json_path, self.json_save_path)
        with open(self.json_save_path, 'r') as in_file:
            self.label_dict = json.load(in_file)
        self.event_id_list = list(self.label_dict.keys())

        with open (self.event_type_path, 'r') as in_file:
            self.event_type_dict = json.load(in_file)
        return


    def setup_ui(self):
        self.root.title("Road Event Dataset Labeling Tool v2.0")
        self.root.geometry("1280x720")
        self.root.columnconfigure([0, 1], minsize=200, weight=1)
        self.root.rowconfigure(0, minsize=200, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.quit_and_save) 
        self.root.bind("s", self.save_annotations)
        self.root.bind("p", self.load_prev_image)
        self.root.bind("n", self.load_next_image) # either clicking "Next" button or pressing "n" button on the keyboard
        self.root.bind("r", self.redo_bounding_boxes)
        self.root.bind("d", self.remove_current_frame)
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=14)

        # Left frame
        frm_left = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        intro_label = tk.Label(
            frm_left,
            text="An interactive tool to tweak the automatically produced bounding box and provide an event type for road event data",
            wraplength=250  # Adjust this to control line wrapping
        )
        entry_label = tk.Label(frm_left, text="Enter event type here:", font=("Helvetica", 14))
        self.e1 = tk.Entry(frm_left)
        self.progress_label = tk.Label(frm_left, text="", font=("Helvetica", 14, "bold"))
        self.progress_label.config(text='_'.join(self.file_list[self.file_idx].split('/')[-1].split('_')[:4]) + f"\n{(self.file_idx+1):d}/{len(self.file_list)}", 
                                   fg="black")
        self.event_type_labels = [
            tk.Label(frm_left, text="Pothole", font=("Helvetica", 12)),
            tk.Label(frm_left, text="Manhole Cover", font=("Helvetica", 12)),
            tk.Label(frm_left, text="Drain Gate", font=("Helvetica", 12)),
            tk.Label(frm_left, text="Unknown", font=("Helvetica", 12)),
            tk.Label(frm_left, text="Speed Bump", font=("Helvetica", 12))
        ]
        self.event_type_labels_locs = [[10, 260], [100, 260], [10, 290], [100, 290], [10, 320]]
        for idx, label in enumerate(self.event_type_labels):
            label.bind("<Button-1>", lambda event, idx=idx: self.select_event_type(event, idx))
            label.place(x=self.event_type_labels_locs[idx][0], y=self.event_type_labels_locs[idx][1])

        self.prev_button = tk.Button(
            frm_left, text="Prev"
        )
        self.prev_button.bind("<Button-1>", self.load_prev_image)
        self.next_button = tk.Button(
            frm_left, text="Next"
        )
        self.save_button = tk.Button(
            frm_left, text="Save"
        )
        self.save_button.bind("<Button-1>", self.save_annotations)
        self.next_button.bind("<Button-1>", self.load_next_image)
        self.redo_button = tk.Button(
            frm_left, text="Redo"
        )
        self.redo_button.bind("<Button-1>", self.redo_bounding_boxes)
        self.remove_button = tk.Button(
            frm_left, text="Remove Frame"
        )
        self.remove_button.bind("<Button-1>", self.remove_current_frame)

        self.message_label = tk.Label(frm_left, text="", fg="green", wraplength=180, font=("Helvetica", 12))
        self.message_label.place(x=10, y=490)

        event_id = "_".join(["s", self.file_list[self.file_idx].split('/')[-1].split('_')[1], 
                             self.file_list[self.file_idx].split('/')[-1].split('_')[2]])
        if self.label_dict[event_id]['wheel']=="3":
            wheel_text = "left"
        elif self.label_dict[event_id]['wheel']=="4":
            wheel_text = "right"
        else:
            wheel_text = "left & right"
        self.wheel_label = tk.Label(frm_left, text=wheel_text, wraplength=180, font=("Helvetica", 18))
        self.wheel_label.place(x=10, y=590)

        # Right frame
        frm_right = tk.Frame(
            self.root, width=800, height=800, relief=tk.RAISED, bd=2
        )
        self.canvas = tk.Canvas(
            frm_right, width=800, height=500, cursor="cross"
        )
        self.canvas.pack()
        self.canvas_spec = tk.Canvas(
            frm_right, width=800, height=300, cursor="cross"
        )
        self.canvas_spec.pack() # the order of packing the widgets determines their stacking order

        intro_label.pack()  # Place the introduction label at the top
        entry_label.pack()
        self.e1.pack()
        self.progress_label.pack()
        # or use pack() with padx and pady
        self.prev_button.place(x=10, y=360)
        self.next_button.place(x=100,y=360)
        self.save_button.place(x=10, y=400)
        self.redo_button.place(x=100, y=400)
        self.remove_button.place(x=10, y=440)

        frm_left.grid(row=0, column=0, sticky="ns")
        frm_right.grid(row=0, column=1, sticky="nsew")

        # read and create an image item on `self.canvas`
        demo_pic = Image.open(self.file_list[self.file_idx])
        demo_pic_resized = demo_pic.resize((self.frame_size[0], self.frame_size[1]), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.finish_polygon)

        # 9.8: add a canvas below the image window to show the wheelAccel segment, print the peak speed in the window and speed at the event timestamp
        # read and create a spec image item on `self.canvas_spec`
        wheelAccel_path = os.path.join(self.data_path, "wheelAccel_seg", event_id+".npy")
        self.figure = Figure(figsize=(8,3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        sensor_data = np.load(wheelAccel_path)[0]
        peaks, _ = find_peaks(sensor_data, prominence=1)
        peaks_2, _ = find_peaks(-sensor_data, prominence=1)
        self.ax.plot(peaks, sensor_data[peaks], "or")
        self.ax.plot(peaks_2, sensor_data[peaks_2], "og")
        self.ax.plot(sensor_data, color='blue', linewidth=2)
        # create a FigureCanvas to embed the matplotlib figure
        self.canvas_fig = FigureCanvasTkAgg(self.figure, master=self.canvas_spec)
        self.canvas_fig.get_tk_widget().pack()


    def select_event_type(self, event, event_index):
        for idx, label in enumerate(self.event_type_labels):
            if idx == event_index:
                label.config(fg="green")
                self.event_label = str(event_index)
            else:
                label.config(fg="black")


    def on_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        # Append the point to the points list
        self.points.append((x, y))
        # Draw line segments between polygon vertex clicks
        if len(self.points) >= 2:
            last_point = self.points[-2]
            self.canvas.create_line(last_point[0], last_point[1], x, y, fill="blue", width=2, tags="bbox")
        
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red", tags="dots")


    def finish_polygon(self, event):
        if len(self.points) >= 3:
            self.canvas.create_polygon(
                [coord for point in self.points for coord in point],
                outline="red",
                width=2,
                fill="",
                tags="bbox",
            )
            self.bbox_list.append(self.points) # get the series of points clicked in a bbox
            if len(self.e1.get()) > 0:
                self.event_label = self.e1.get() 
        self.points = []


    def redo_bounding_boxes(self, event):
        self.canvas.delete("bbox")
        self.canvas.delete("dots")
        self.bbox_list = []
        self.points = []
        self.message_label.config(text="", fg="green")


    def load_next_image(self, event):
        self.file_idx = (self.file_idx + 1) % len(self.file_list)
        self.update_image()


    def load_prev_image(self, event):
        self.file_idx = (self.file_idx - 1) % len(self.file_list)
        self.update_image()


    def update_image(self):
        # open and display the new image
        demo_pic = Image.open(self.file_list[self.file_idx])
        demo_pic_resized = demo_pic.resize((self.frame_size[0], self.frame_size[1]), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)

        # open and display the new wheelAccel
        event_id = "_".join(["s", self.file_list[self.file_idx].split('/')[-1].split('_')[1], 
                             self.file_list[self.file_idx].split('/')[-1].split('_')[2]])
        wheelAccel_path = os.path.join(self.data_path, "wheelAccel_seg", event_id+".npy")
        self.ax.clear()
        sensor_data = np.load(wheelAccel_path)[0]
        peaks, _ = find_peaks(sensor_data, prominence=1)
        peaks_2, _ = find_peaks(-sensor_data, prominence=1)
        self.ax.plot(peaks, sensor_data[peaks], "or")
        self.ax.plot(peaks_2, sensor_data[peaks_2], "og")
        self.ax.plot(sensor_data, color='blue', linewidth=2)
        self.canvas_fig.draw()

        self.canvas.delete("bbox") # clear previous bounding boxes
        self.bbox_list = [] # clear previous bounding box data
        self.points = [] # clear previous points
        self.e1.delete(0, tk.END) # update the entry field to clear the previous event type
        for idx, label in enumerate(self.event_type_labels):
            label.config(fg="black")
        self.progress_label.config(text='_'.join(self.file_list[self.file_idx].split('/')[-1].split('_')[:4]) + f"\n{(self.file_idx+1):d}/{len(self.file_list)}", 
                                   fg="black")
        
        if self.label_dict[event_id]['wheel']=="3":
            wheel_text = "left"
        elif self.label_dict[event_id]['wheel']=="4":
            wheel_text = "right"
        else:
            wheel_text = "left & right"
        self.wheel_label.config(text=wheel_text, 
                                fg="black")
    

    def remove_current_frame(self, event):
        event_id = '_'.join(self.file_list[self.file_idx].split('/')[-1].split('_')[:4])
        frame_idx = int(self.file_list[self.file_idx].split('/')[-1].split('_')[5])
        assert (event_id in self.event_id_list), "Queried event is not in the original metafile!"
        # remove current frame and its bbox coords from self.label_dict['data']
        del self.label_dict['data'][event_id]['frame_paths'][frame_idx-(self.frames_per_event-len(self.label_dict['data'][event_id]['frame_paths']))]
        del self.label_dict['data'][event_id]['bbox_coords'][frame_idx-(self.frames_per_event-len(self.label_dict['data'][event_id]['frame_paths']))]
        if len(self.label_dict['data'][event_id]['frame_paths']) == 0:
            del self.label_dict['data'][event_id]
        # remove current frame from self.file_list too, and reverse the file index to load the next frame in the updated self.file_list
        del self.file_list[self.file_idx]
        self.file_idx -= 1
        # load the new image
        self.load_next_image(event)


    def save_annotations(self, event):
        # save self.event_label, and self.bbox_list
        event_id = '_'.join(self.file_list[self.file_idx].split('/')[-1].split('_')[:4])
        frame_idx = int(self.file_list[self.file_idx].split('/')[-1].split('_')[3])
        assert (event_id in self.event_id_list), "Queried event is not in the original metafile!"

        if len(self.e1.get()) > 0:
            self.event_label = self.e1.get()
        else:
            pass
        # make changes to the copy of meta json file if needed 
        if len(self.event_label) > 0:
            self.label_dict['data'][event_id]['event_label'] = self.event_label
            self.label_dict['data'][event_id]['event_type'] = self.event_type_dict[self.event_label]
            self.message_label.config(text=f"{event_id}_frame_{frame_idx:d} labels changed and saved", fg="green")
        # allowing single event bbox for each frame, for now
        if len(self.bbox_list) > 0:
            self.bbox_list[0][1], self.bbox_list[0][3] = self.bbox_list[0][3], self.bbox_list[0][1]
            self.bbox_list[0][2], self.bbox_list[0][3] = self.bbox_list[0][3], self.bbox_list[0][2]
            self.label_dict['data'][event_id]['bbox_coords'][frame_idx] = (np.array(self.bbox_list[0])*np.array([1920/self.frame_size[0], 1080/self.frame_size[1]])).tolist()
            self.message_label.config(text=f"{event_id}_frame_{frame_idx:d} labels changed and saved", fg="green")
        print(self.label_dict['data'][event_id])
        return
        
    
    def save_label_data(self):
        with open(self.json_save_path, 'w', encoding='utf-8') as outfile:
            json.dump(self.label_dict, outfile)
        print('Manually labeled data saved!')
    

    def quit_and_save(self):
        self.save_label_data()
        self.root.destroy()
    

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder-path', type=str, default='', required=True, help='data folder path')
    parser.add_argument('-p', '--prev-json-path', type=str, default='', required=True, help='path to previously saved meta json file')
    parser.add_argument('-e', '--event-type-path', type=str, default='', required=True, help='path to event label to event type convertion')
    parser.add_argument('-s', '--json-save-path', type=str, default='', required=True, help='path to save the manually-labled meta json file')
    parser.add_argument('-f', '--frame-size', type=int, nargs='+', required=True, help='the frame size of image to edit')
    parser.add_argument('-n', '--frames-per-event', type=int, default=3, required=True, help='number of frames per event in the current folder')
    args = parser.parse_args()
    # args.json_save_path = os.path.join('/'.join(args.prev_json_path.split('/')[:-1]), f"events_metafile_manually_labeled_{args.data_folder_path.split('/')[-1].split('_')[-1]}.json")

    root = tk.Tk()
    app = BoundingBoxApp(root, 
                         data_path=args.data_folder_path,
                         event_type_path=args.event_type_path,
                         prev_path=args.prev_json_path,
                         save_path=args.json_save_path,
                         image_frame_size=args.frame_size,
                         frames_per_event=args.frames_per_event)
    root.mainloop()
    

if __name__ == "__main__":
    main()

# TO-DOs:
    # copy the original meta json file and make changes to the copy
    # print event label to event type convertion in text label in tkinter
    # print the current event_id and frame_id in tkinter
    # add functionality to visualize the manually labeled bbox in a new folder under data path
