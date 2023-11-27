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
from tkinter.simpledialog import askfloat
import pdb # for debug

#%% BoundingBoxApp v1
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
        dataFolderNames = [s for s in os.listdir(self.data_path) \
                        if (os.path.isdir(os.path.join(self.data_path, s)) and s.split('_')[-1].isnumeric())]
        self.sessionId_list = [s.split('_')[-1] for s in dataFolderNames]
        self.file_list = glob.glob(os.path.join(self.data_path, "results", "frames_rv_annotated", "*.png"))
        self.file_list = natsorted(self.file_list)
        return 
    

    def prepare_json(self):
        shutil.copy(self.prev_json_path, self.json_save_path)
        with open(self.json_save_path, 'r') as in_file:
            self.label_dict = json.load(in_file)
        self.event_id_list = list(self.label_dict['data'].keys())

        with open (self.event_type_path, 'r') as in_file:
            self.event_type_dict = json.load(in_file)
        return


    def setup_ui(self):
        self.root.title("Road Event Dataset Labeling Tool v1.0")
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
            wraplength=180  # Adjust this to control line wrapping
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

        # Right frame
        frm_right = tk.Frame(
            self.root, width=600, height=400, relief=tk.RAISED, bd=2
        )
        self.canvas = tk.Canvas(
            frm_right, width=800, height=600, cursor="cross"
        )
        self.canvas.pack()

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

        demo_pic = Image.open(self.file_list[self.file_idx])
        demo_pic_resized = demo_pic.resize((self.frame_size[0], self.frame_size[1]), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.finish_polygon)


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
        self.canvas.delete("bbox") # clear previous bounding boxes
        self.bbox_list = [] # clear previous bounding box data
        self.points = [] # clear previous points
        self.e1.delete(0, tk.END) # update the entry field to clear the previous event type
        for idx, label in enumerate(self.event_type_labels):
            label.config(fg="black")
        self.progress_label.config(text='_'.join(self.file_list[self.file_idx].split('/')[-1].split('_')[:4]) + f"\n{(self.file_idx+1):d}/{len(self.file_list)}", 
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
        frame_idx = int(self.file_list[self.file_idx].split('/')[-1].split('_')[5])
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


# def main():
#     # Arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--data-folder-path', type=str, default='', required=True, help='data folder path')
#     parser.add_argument('-p', '--prev-json-path', type=str, default='', required=True, help='path to previously saved meta json file')
#     parser.add_argument('-e', '--event-type-path', type=str, default='', required=True, help='path to event label to event type convertion')
#     parser.add_argument('-s', '--json-save-path', type=str, default='', required=True, help='path to save the manually-labled meta json file')
#     parser.add_argument('-f', '--frame-size', type=int, nargs='+', required=True, help='the frame size of image to edit')
#     parser.add_argument('-n', '--frames-per-event', type=int, default=3, required=True, help='number of frames per event in the current folder')
#     args = parser.parse_args()
#     # args.json_save_path = os.path.join('/'.join(args.prev_json_path.split('/')[:-1]), f"events_metafile_manually_labeled_{args.data_folder_path.split('/')[-1].split('_')[-1]}.json")

#     root = tk.Tk()
#     app = BoundingBoxApp(root, 
#                          data_path=args.data_folder_path,
#                          event_type_path=args.event_type_path,
#                          prev_path=args.prev_json_path,
#                          save_path=args.json_save_path,
#                          image_frame_size=args.frame_size,
#                          frames_per_event=args.frames_per_event)
#     root.mainloop()
    

# if __name__ == "__main__":
#     main()

# TO-DOs:
    # copy the original meta json file and make changes to the copy
    # print event label to event type convertion in text label in tkinter
    # print the current event_id and frame_id in tkinter
    # add functionality to visualize the manually labeled bbox in a new folder under data path


#%% BoundingBoxApp v2
class BoundingBoxAppv2:
    def __init__(self, root, image_path, wheelAccel_path, prev_json_path, save_json_path, event_type_path, frame_size):
        self.root = root
        self.image_path = image_path
        self.wheelAccel_path = wheelAccel_path
        self.prev_json_path = prev_json_path
        self.save_json_path = save_json_path
        with open(event_type_path, 'r') as in_file:
            self.event_type_dict = json.load(in_file)
        self.prepare_json()
        try:
            self.event_idx, self.frame_idx = int(self.prev_json_path.split('_')[-2]), \
                                             int(self.prev_json_path.split('_')[-1].split('.')[0])
        except:
            self.event_idx = 0
            self.frame_idx = 0
        self.fs = frame_size
        self.rect = None
        self.oval = None
        self.setup_ui()


    def prepare_json(self):
        shutil.copy(self.prev_json_path, self.save_json_path)
        with open(self.save_json_path, 'r') as in_file:
            self.event_dict = json.load(in_file)
        self.event_list = list(self.event_dict.keys())


    def setup_ui(self):
        # root and configs
        self.root.title("Road Event Dataset Labeling Tool v2.0")
        self.root.geometry("1280x720")
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=4)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=2)
        self.root.rowconfigure(2, weight=3)
        intro_font = ("Helvetica", 10, "bold")
        label_font = ("Helvetica", 14, "bold")

        # bind keys
        self.root.protocol("WM_DELETE_WINDOW", self.quit_and_save)
        self.root.bind("n", self.next_image)
        self.root.bind("p", self.prev_image)
        self.root.bind("d", self.add_difficult_label)

        # frames
        frm_left_0 = tk.Frame(self.root, relief=tk.RAISED, bd=2, bg="white")
        frm_left_0.grid(row=0, column=0, sticky='WENS') 
        frm_left_1 = tk.Frame(self.root, relief=tk.RAISED, bd=2, bg="white")
        frm_left_1.grid(row=1, column=0, sticky="WENS")
        frm_left_2 = tk.Frame(self.root, relief=tk.RAISED, bd=2, bg="white")
        frm_left_2.grid(row=2, column=0, sticky="WENS")

        
        frm_right = tk.Frame(self.root, relief=tk.RAISED, bd=2, bg="white")
        frm_right.grid(row=0, column=1, rowspan=3,sticky='WENS')

        # widgets
        l_intro = tk.Label(frm_left_0,
                           text="An UI tool to drag and reposition the bounding box and to provide an event type",
                           font=intro_font,
                           wraplength=150)
        l_intro.place(x=0, y=0)
        self.l_progress = tk.Label(frm_left_1,
                                   text=self.event_dict[self.event_list[self.event_idx]]['frames'][self.frame_idx]+'\n\n'+f"Event {self.event_idx}/{len(self.event_dict)}",
                                   font=label_font,
                                   wraplength=150)
        self.l_progress.place(x=0, y=0)
        self.next_button = tk.Button(frm_left_1, 
                                     text="Next",
                                     font=label_font)
        self.next_button.place(x=0, y=120)
        self.prev_button = tk.Button(frm_left_1,
                                     text="Prev",
                                     font=label_font)
        self.prev_button.place(x=80, y=120)
        # button to indicate difficult sample
        self.diff_button = tk.Button(frm_left_2,
                                     text="Difficult",
                                     font=label_font)
        self.diff_button.place(x=0, y=0)
        self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx] == 0
        fg = "green"
        self.l_diff = tk.Label(frm_left_2,
                               text=f"Current frame has difficult label {self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx]}",
                               font=intro_font,
                               wraplength=150,
                               fg=fg)
        self.l_diff.place(x=0, y=80)
               

        # canvas
        self.canvas = tk.Canvas(frm_right, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        demo_pic = Image.open(os.path.join(self.image_path, self.event_dict[self.event_list[self.event_idx]]['frames'][self.frame_idx]+'.png'))
        demo_pic_resized = demo_pic.resize((self.fs[0], self.fs[1]), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)

        # extract rv_rot_rect_box coordinates
        self.bbox_coords = np.array(self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box"][self.frame_idx])
        self.bbox_coords *= ([self.fs[0]/1920, self.fs[1]/1080])
        self.bbox_coords = np.float_(self.bbox_coords) # br, tr, tl, bl
        self.center_coords = np.array(self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box_dim"][self.frame_idx][:2])
        self.center_coords *= ([self.fs[0]/1920, self.fs[1]/1080])
        self.draw_rectangle()

        # more bind keys
        self.next_button.bind("<Button-1>", self.next_image)
        self.prev_button.bind("<Button-1>", self.prev_image)
        self.diff_button.bind("<Button-1>", self.add_difficult_label)
        self.canvas.bind("<Button-1>", self.on_click)
        # self.canvas.bind("<B1-Motion>", self.on_drag)
        # self.root.bind("<s>", self.scale_rectangle)


    def draw_rectangle(self):
        if self.rect:
            self.canvas.delete(self.rect)
        if self.oval:
            self.canvas.delete(self.oval)
        self.rect = self.canvas.create_polygon(*self.bbox_coords.flatten(), outline="red", fill='') # asterisk to unpack
        self.oval = self.canvas.create_oval(self.center_coords[0] - 2,
                                            self.center_coords[1] - 2,
                                            self.center_coords[0] + 2,
                                            self.center_coords[1] + 2,
                                            fill="red")

    def on_click(self, event):
        self.new_x, self.new_y = event.x, event.y # integers
        dx, dy = self.new_x - self.bbox_coords[2][0], \
                 self.new_y - self.bbox_coords[2][1]
        # update self.bbox_coords
        self.bbox_coords[:,0] += dx
        self.bbox_coords[:,1] += dy
        self.center_coords[0] += dx
        self.center_coords[1] += dy
        self.draw_rectangle()

        # save the new bbox coords in the dictionary
        rv_bbox_coords = self.bbox_coords * np.array([1920/self.fs[0], 1080/self.fs[1]])
        self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box"][self.frame_idx] = rv_bbox_coords.tolist()
        
        # save updated bbox dims in the dictionary (the box center changed)
        # "rv_rot_rect_box_dim" (x_c, y_c, yaw, w, h, a)
        self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box_dim"][self.frame_idx][0] = self.center_coords[0]*(1920/self.fs[0])
        self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box_dim"][self.frame_idx][1] = self.center_coords[1]*(1080/self.fs[1])


    def next_image(self, event):
        self.canvas.delete("all")
        if self.frame_idx == len(self.event_dict[self.event_list[self.event_idx]]["frames"]) - 1:
            self.frame_idx = 0
            self.event_idx += 1
            if self.event_idx == len(self.event_dict): # circular
                self.event_idx = 0
        else:
            self.frame_idx += 1    

        self.l_progress.config(text=self.event_dict[self.event_list[self.event_idx]]['frames'][self.frame_idx]+'\n\n'+f"Event {self.event_idx}/{len(self.event_dict)}")
        if self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx] == 0: fg = "green"
        else: fg = "red"
        self.l_diff.config(text=f"Current frame has difficult label {self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx]}",
                           fg=fg)

        demo_pic = Image.open(os.path.join(self.image_path, self.event_dict[self.event_list[self.event_idx]]['frames'][self.frame_idx]+'.png'))
        demo_pic_resized = demo_pic.resize((self.fs[0], self.fs[1]), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)
        
        self.bbox_coords = np.array(self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box"][self.frame_idx])
        self.bbox_coords *= ([self.fs[0]/1920, self.fs[1]/1080])
        self.bbox_coords = np.float_(self.bbox_coords) # br, tr, tl, bl
        self.center_coords = np.array(self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box_dim"][self.frame_idx][:2])
        self.center_coords *= ([self.fs[0]/1920, self.fs[1]/1080])
        self.draw_rectangle()


    def prev_image(self, event):
        self.canvas.delete("all")
        # update event_idx first, if currently the first image
        if self.frame_idx == 0:
            if self.event_idx == 0:
                self.event_idx = len(self.event_dict) - 1
            else:
                self.event_idx -= 1
            len_frames = len(self.event_dict[self.event_list[self.event_idx]]['frames'])
            self.frame_idx = len_frames - 1
        else:
            self.frame_idx -= 1
        self.l_progress.config(text=self.event_dict[self.event_list[self.event_idx]]['frames'][self.frame_idx]+'\n\n'+f"Event {self.event_idx}/{len(self.event_dict)}")
        if self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx] == 0: fg = "green"
        else: fg = "red"
        self.l_diff.config(text=f"Current frame has difficult label {self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx]}",
                           fg=fg)

        demo_pic = Image.open(os.path.join(self.image_path, self.event_dict[self.event_list[self.event_idx]]['frames'][self.frame_idx]+'.png'))
        demo_pic_resized = demo_pic.resize((self.fs[0], self.fs[1]), Image.Resampling.LANCZOS)
        self.demo_img = ImageTk.PhotoImage(demo_pic_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.demo_img)
        
        self.bbox_coords = np.array(self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box"][self.frame_idx])
        self.bbox_coords *= ([self.fs[0]/1920, self.fs[1]/1080])
        self.bbox_coords = np.float_(self.bbox_coords) # br, tr, tl, bl
        self.center_coords = np.array(self.event_dict[self.event_list[self.event_idx]]["rv_rot_rect_box_dim"][self.frame_idx][:2])
        self.center_coords *= ([self.fs[0]/1920, self.fs[1]/1080])
        self.draw_rectangle()

    
    def add_difficult_label(self, event):
        # update the difficult label
        self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx] = 1
        self.l_diff.config(text=f"Current frame has difficult label {self.event_dict[self.event_list[self.event_idx]]['difficult'][self.frame_idx]}",
                           fg="red")


    def save_label_data(self):
        # save current self.event_idx, self.frame_idx in the filename. so that editting can happen from the current event_idx next time
        with open(self.save_json_path.split(".")[0]+f"_{self.event_idx}_{self.frame_idx}.json", 'w', encoding='utf-8') as outfile:
            json.dump(self.event_dict, outfile)
        print('Manually labeled data saved!')
        

    def quit_and_save(self):
        self.save_label_data()
        self.root.destroy()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder-path', type=str, default='', required=True, help='data folder path')
    parser.add_argument('-p', '--prev-json-path', type=str, default='', required=True, help='previous json path')
    parser.add_argument('-e', '--event-type-path', type=str, default='', required=True, help='event class taxonomy path')
    parser.add_argument('-s', '--save-json-path', type=str, default='', required=True, help='path to save updated json')
    parser.add_argument('-f', '--frame-size', type=int, nargs='+', required=True, help='tkinter canvas frame size')
    args = parser.parse_args()

    # Tkinter init
    # pdb.set_trace()
    root = tk.Tk()
    app = BoundingBoxAppv2(
                           root, 
                           image_path=os.path.join(args.data_folder_path, "undistorted_rv"),
                           wheelAccel_path=os.path.join(args.data_folder_path, "wheelAccel_seg"),
                           prev_json_path=args.prev_json_path,
                           save_json_path=args.save_json_path,
                           event_type_path=args.event_type_path,
                           frame_size=args.frame_size
                           )
    root.mainloop()
# %%
