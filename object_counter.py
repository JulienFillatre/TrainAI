# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.count_ids_down = []
        self.count_ids_up = []
        self.track_ids_in = []
        self.track_ids_out = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 255, 255)
        self.count_bg_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.track_color = None

    def set_args(
        self,
        deltaHeightline,
        classes_names,
        heightLine,
        reg_pts,
        count_reg_color=(255, 0, 255),
        count_txt_color=(255, 255, 255),
        count_bg_color=(0, 0, 0),
        line_thickness=2,
        track_thickness=2,
        view_in_counts=True,
        view_out_counts=True,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_color (RGB color): count text color value
            count_bg_color (RGB color): count highlighter line color
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            cls_txtdisplay_gap (int): Display gap between each class count
        """
        self.tf = line_thickness
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.heightLine = heightLine
        self.deltaHeightLine = deltaHeightline

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.cls_txtdisplay_gap = cls_txtdisplay_gap

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                # Draw bounding box
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    if len(self.names[cls]) > 5:
                        self.names[cls] = self.names[cls][:5]
                    self.class_wise_count[self.names[cls]] = {"in": 0, "out": 0}

                # Count objects using line
                if len(self.reg_pts) == 2:
                    if(track_id not in self.count_ids_up and track_id not in self.count_ids_down):
                        if int(y) >= self.heightLine + self.deltaHeightLine : 
                            self.count_ids_down.append(track_id)
                        if int(y) <= self.heightLine - self.deltaHeightLine: 
                            self.count_ids_up.append(track_id)
                    
                    #Condition dans une liste présente 
                    if(track_id in self.count_ids_up and track_id not in self.count_ids_down):
                        if int(y) >= self.heightLine + self.deltaHeightLine :
                            if track_id not in self.track_ids_out : 
                                self.count_ids_up.remove(track_id)
                                self.track_ids_out.append(track_id) 
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["out"] += 1


                    if(track_id not in self.count_ids_up and track_id in self.count_ids_down):
                        if int(y) <= self.heightLine - self.deltaHeightLine :
                            if track_id not in self.track_ids_in : 
                                self.count_ids_down.remove(track_id)
                                self.track_ids_in.append(track_id) 
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["in"] += 1
                             
                        

                            
                            
                            #recup hauteur ligne
                            #si id dans partie basse et déjà ajouté dans partie haute alors out 
                            #si id dans partie basse et non dans partie haute ajout id dans liste_partie-basse
                            #si id dans partie haute et déjà présent dans partie basse alors in 
                            #si id dans partie haute et non dans partie basse alors ajout id liste_partie-haute 
                            #Créer deux cotés, si track_id a up alors track_id.append(track_id_up)
                            #Si track id dans track_id-up, verifie si track_id dans track_id_down
                                #Si dans 


        label = ""

        for key, value in self.class_wise_count.items():
            if value["in"] != 0 or value["out"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    label = None
                elif not self.view_in_counts:
                    label += f"Entree {value['in']} \t"
                elif not self.view_out_counts:
                    label += f"Sortie {value['out']} \t"
                else:
                    label += f"Entree : {value['in']}, Sortie : {value['out']}, Total Interieur : {value['in']-value['out']} \t"

        label = label.rstrip()
        label = label.split("\t")

        if label is not None:
            self.annotator.display_counts(
                counts=label,
                count_txt_color=self.count_txt_color,
                count_bg_color=self.count_bg_color,
            )

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects
        return self.im0


if __name__ == "__main__":
    ObjectCounter()
