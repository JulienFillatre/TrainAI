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
        st_average_inference = st.empty()
        st_frames = st.empty()
        st_resolution = st.empty()
        inference_times = []
        isFirstFrame=0
        frame_counter=0

        try:
            options = {"STREAM_RESOLUTION": f'{resolutionVideo}p', "STREAM_PARAMS": {"nocheckcertificate": True}}
            cap = CamGear(source=sourceVideo, stream_mode=isYoutube, logging=True, **options).start()  # YouTube Video URL as input
            while True:
                originalImage = cap.read()
                if originalImage is None:
                    break
                
                originalImage2 = originalImage.copy()
                cleanedImage = cv2.convertScaleAbs(originalImage, alpha=contrast, beta=brightness)
                
                start_time = time.time()
                res = myModel.track(cleanedImage, conf=myConf, persist=True, classes=myClasses, iou=iou)
                inference_time = (time.time() - start_time)*1000
                inference_times.append(inference_time)
                res_plotted = res[0].plot()
                frame_counter += 1
                if frame_counter % 30 == 0:
                    average_inference_time = sum(inference_times) / len(inference_times)
                    st_average_inference.write(f"Temps moyen de traitement par image : {average_inference_time:.1f} ms")
                    st_frames.write(f"Nombre d'images traitées : {frame_counter}")
                if isFirstFrame == 0:
                    st_resolution.write(f"Résolution choisie par le backend yt-dlp : {originalImage.shape}")
                    isFirstFrame = 1
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
    heightLine = st.slider('Hauteur de ligne de comptage (px)', 0, int(resolutionVideo), int(resolutionVideo/2))
    deltaHeightLine = st.slider('Δ Comptage (px)', 0, 50, 25)
    line_points = [(0, resolutionVideo-heightLine), (wideImage, resolutionVideo-heightLine)]
    if st.sidebar.button('Lancer la détection'):
        st_frame = st.empty()
        if isDisplayOriginal:
            st_frame2 = st.empty()
        st_average_inference = st.empty()
        st_frames = st.empty()
        st_resolution = st.empty()
        isFirstFrame = 0
        frame_counter = 0
        inference_times = []
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
                inference_time = (time.time() - start_time)*1000
                inference_times.append(inference_time)
                frame_counter += 1
                if frame_counter % 30 == 0:
                    average_inference_time = sum(inference_times) / len(inference_times)
                    st_average_inference.write(f"Temps moyen de traitement par image : {average_inference_time:.1f} ms")
                    st_frames.write(f"Nombre d'images traitées : {frame_counter}")
                if isFirstFrame == 0:
                    st_resolution.write(f"Résolution choisie par le backend yt-dlp : {originalImage.shape}")
                    isFirstFrame = 1
                
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