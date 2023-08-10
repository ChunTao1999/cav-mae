import argparse
import os
import tkinter as tk
import glob
from PIL import Image, ImageTk
from natsort import natsorted
import numpy as np
import json
import pdb # for debug


class BoundingBoxApp:
    def __init__(self, root, folder_path, save_path):
        self.root = root
        self.folder = folder_path
        self.json_save_path = save_path
        self.file_list = glob.glob(os.path.join(self.folder, '*.png'))
        self.file_list = natsorted(self.file_list)
        self.file_idx = 0
        self.canvas = None
        self.points = []
        self.event_label = ""
        self.bbox_list = []
        # Here, instead of intiating a blank dictionary, can copy the auto-generated dict and manually tweak the dataset metadata on top of it.
        self.label_dict = {}
        self.label_dict['label_data'] = []
        self.setup_ui()


    def setup_ui(self):
        self.root.title("Road Event Dataset Labeling Tool v1.0")
        self.root.geometry("1280x720")
        self.root.columnconfigure([0, 1], minsize=200, weight=1)
        self.root.rowconfigure(0, minsize=200, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.quit_and_save) 
        self.root.bind("s", self.save_annotations)
        self.root.bind("n", self.load_next_image) # either clicking "Next" button or pressing "n" button on the keyboard
        self.root.bind("r", self.redo_bounding_boxes)

        frm_left = tk.Frame(self.root, relief=tk.RAISED, bd=2)
        
        # Add program introduction label
        intro_label = tk.Label(
            frm_left,
            text="An interactive tool to tweak the automatically produced bounding box and provide an event type for road event data",
            wraplength=180  # Adjust this to control line wrapping
        )
        
        entry_label = tk.Label(frm_left, text="Enter event type here:")
        self.e1 = tk.Entry(frm_left)
        self.save_button = tk.Button(
            frm_left, text="Save"
        )
        self.save_button.bind("<Button-1>", self.save_annotations)
        self.next_button = tk.Button(
            frm_left, text="Next"
        )
        self.next_button.bind("<Button-1>", self.load_next_image)
        self.redo_button = tk.Button(
            frm_left, text="Redo"
        )
        self.redo_button.bind("<Button-1>", self.redo_bounding_boxes)
        
        frm_right = tk.Frame(
            self.root, width=600, height=400, relief=tk.RAISED, bd=2
        )

        self.canvas = tk.Canvas(
            frm_right, width=640, height=480, cursor="cross"
        )
        self.canvas.pack()

        intro_label.pack()  # Place the introduction label at the top
        entry_label.pack()
        self.e1.pack()
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.next_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.redo_button.pack(side=tk.LEFT, padx=5, pady=5)

        frm_left.grid(row=0, column=0, sticky="ns")
        frm_right.grid(row=0, column=1, sticky="nsew")

        demo_pic = Image.open(self.file_list[self.file_idx])
        demo_pic_resized = demo_pic.resize((640, 480), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.finish_polygon)


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
            self.event_label = self.e1.get()
        self.points = []


    def redo_bounding_boxes(self, event):
        self.canvas.delete("bbox")
        self.canvas.delete("dots")
        self.bbox_list = []
        self.points = []


    def load_next_image(self, event):
        self.file_idx = (self.file_idx + 1) % len(self.file_list)
        self.update_image()


    def update_image(self):
        # open and display the new image
        demo_pic = Image.open(self.file_list[self.file_idx])
        demo_pic_resized = demo_pic.resize((640, 480), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)
        self.canvas.delete("bbox") # clear previous bounding boxes
        self.bbox_list = [] # clear previous bounding box data

        self.e1.delete(0, tk.END) # update the entry field to clear the previous event type


    def save_annotations(self, event):
        # save self.event_label, and self.bbox_list
        if len(self.label_dict['label_data']) > self.file_idx:
            self.label_dict['label_data'].pop()
        self.event_label = self.e1.get()
        self.label_dict['label_data'].append({'frame_path': self.file_list[self.file_idx],
                                              'event_label': self.event_label,
                                              'bbox_list': self.bbox_list})
        
    
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
    parser.add_argument('-n', '--image-file-name', type=str, default='', required=True, help='name of the frame to load')
    parser.add_argument('-s', '--json-save-path', type=str, default='', required=True, help='path to save the resulting label json file')
    args = parser.parse_args()

    root = tk.Tk()
    app = BoundingBoxApp(root, 
                         folder_path=args.data_folder_path, 
                         save_path=args.json_save_path)
    root.mainloop()


if __name__ == "__main__":
    main()
