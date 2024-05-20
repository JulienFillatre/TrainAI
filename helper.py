import object_counter
import streamlit as st
import cv2
from vidgear.gears import CamGear
import time

def visualize(myConf, myModel, myClasses, iou, sourceVideo, isYoutube, isDisplayOriginal, resolutionVideo, contrast, brightness):
    if st.sidebar.button('Lancer la détection'):
        st_frame = st.empty()
        if isDisplayOriginal:
            st_frame2 = st.empty()
        st_inference = st.empty()
        try:
            options = {"STREAM_RESOLUTION": f'{resolutionVideo}p', "STREAM_PARAMS": {"nocheckcertificate": True}}
            cap = CamGear(source=sourceVideo, stream_mode=isYoutube, logging=True, **options).start()  # YouTube Video URL as input
            while True:
                originalImage = cap.read()
                originalImage2 = originalImage
                cleanedImage = cv2.convertScaleAbs(originalImage, alpha=contrast, beta=brightness)
                start_time = time.time()
                res = myModel.track(cleanedImage, conf=myConf, persist=True, classes=myClasses, iou=iou)
                res_plotted = res[0].plot()
                inference_time = time.time() - start_time
                    # Afficher la moyenne du temps d'inférence sur l'interface Streamlit
                st_inference.write(f"Temps d'inférence : {inference_time:.4f} secondes")
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               width=1280
                               )
                if isDisplayOriginal:
                    st_frame2.image(originalImage2,
                                    caption='Original Video',
                                    channels="BGR",
                                    width=1280
                                    )
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        finally:
            if cap is not None:
                cap.stop()
                st_frame.empty()
                if isDisplayOriginal:
                    st_frame2.empty()


def counter(myConf, myModel, myClasses, iou, sourceVideo, isYoutube, isDisplayOriginal, resolutionVideo, contrast, brightness):
    wideImage = resolutionVideo*16/9
    heightLine = st.slider('Hauteur de ligne de comptage', 0, int(resolutionVideo), int(resolutionVideo/2))
    deltaHeightLine = st.slider('Delta Comptage', 0, 50, 25)
    #heightLinePointLeft = st.slider('Hauteur de ligne à gauche', 0, int(resolutionVideo), int(resolutionVideo/2))
    line_points = [(0, resolutionVideo-heightLine), (wideImage, resolutionVideo-heightLine)]
    if st.sidebar.button('Lancer la détection'):
        st_frame = st.empty()
        if isDisplayOriginal:
            st_frame2 = st.empty()
        st_inference = st.empty()
        st_resolution = st.empty()
        isFirstFrame = 0
        counter = object_counter.ObjectCounter()
        counter.set_args(
                        reg_pts=line_points,
                        heightLine=resolutionVideo-heightLine,
                        deltaHeightline=deltaHeightLine,
                        line_thickness=2,
                        classes_names=myModel.names,
                        track_thickness=2,
                        track_color=(0, 255, 0),
                        view_in_counts=True,
                        view_out_counts=True,
                        region_thickness=3,
                        count_reg_color=(255, 0, 0),
                        line_dist_thresh=5
                        )
        try:
            options = {"STREAM_RESOLUTION": f'{resolutionVideo}p', "STREAM_PARAMS": {"nocheckcertificate": True}}
            cap = CamGear(source=sourceVideo, stream_mode=isYoutube, logging=True, **options).start()  # YouTube Video URL as input
            while True:
                originalImage = cap.read()
                originalImage2 = originalImage
                cleanedImage = cv2.convertScaleAbs(originalImage, alpha=contrast, beta=brightness)
                start_time = time.time()
                res = myModel.track(cleanedImage, conf=myConf, persist=True, classes=myClasses, iou=iou)
                detectedImage = counter.start_counting(cleanedImage, res)
                inference_time = time.time() - start_time
                st_inference.write(f"Temps d'inférence : {inference_time:.4f} secondes")
                if isFirstFrame == 0:
                    st_resolution.write(f"Résolution Choisie par le backend yt-dlp : {originalImage.shape}")
                    isFirstFrame = 1
                st_frame.image(detectedImage,
                               caption='Detected Video',
                               channels="BGR",
                               width=1280
                               )
                if isDisplayOriginal:
                    st_frame2.image(originalImage2,
                                    caption='Original Video',
                                    channels="BGR",
                                    width=1280
                                    )
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        finally:
            if cap is not None:
                cap.stop()
                st_frame.empty()
                if isDisplayOriginal:
                    st_frame2.empty()