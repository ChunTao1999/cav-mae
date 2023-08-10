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
    def __init__(self, root, folder_path):
        self.root = root
        self.folder = folder_path
        self.file_list = glob.glob(os.path.join(self.folder, '*.png'))
        self.file_list = natsorted(self.file_list)
        self.file_idx = 0
        self.canvas = None
        self.points = []
        self.bbox_list = []
        self.label_dict = {}
        self.label_dict['label_data'] = []
        self.setup_ui()


    def setup_ui(self):
        self.root.title("Road Event Dataset Labeling Tool v1.0")
        self.root.geometry("1280x720")
        self.root.columnconfigure([0, 1], minsize=200, weight=1)
        self.root.rowconfigure(0, minsize=200, weight=1)

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
            frm_left, text="Save", command=self.save_annotations
        )
        self.next_button = tk.Button(
            frm_left, text="Next", command=self.load_next_image
        )
        
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
            self.canvas.create_line(last_point[0], last_point[1], x, y, fill="blue")
        
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")


    def finish_polygon(self):
        if len(self.points) >= 3:
            self.canvas.create_polygon(
                [coord for point in self.points for coord in point],
                outline="red",
                width=2,
                fill="",
                tags="bbox",
            )
            self.bbox_list.append(self.points) # get the series of points clicked in a bbox
            self.event_type = self.e1.get()
        self.points = []


    def load_next_image(self):
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


    def save_annotations(self):
        # save self.event_type, and self.bbox_list
        if len(self.label_dict['label_data']) > self.file_idx:
            self.label_dict['label_data'].pop()
        self.label_dict['label_data'].append({'frame_path': self.file_list[self.file_idx],
                                              'event_type': self.event_type,
                                              'bbox_list': self.bbox_list})
        
    
    def save_label_data(self):
        with open('/home/nano01/a/tao88/RoadEvent-Dataset/tweaked_labels/labels.json', 'w') as outfile:
            json.dump(self.label_dict, outfile)
        print('Manually labeled data saved!')


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder-path', default='', required=True, help='data folder path')
    parser.add_argument('-n', '--image-file-name', default='', required=True, help='name of the frame to load')
    args = parser.parse_args()

    root = tk.Tk()
    app = BoundingBoxApp(root, args.data_folder_path)
    root.mainloop()
    # on exit click, save the label dictionary
    app.save_label_data()


if __name__ == "__main__":
    main()
