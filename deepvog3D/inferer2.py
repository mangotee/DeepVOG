import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import skvideo.io as skv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.draw import line, circle_perimeter
from skimage.transform import resize
from skvideo.utils import rgb2gray
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from .deepvog_torsion.torsion_lib.CrossCorrelation import genPolar, findTorsion
from .deepvog_torsion.torsion_lib.Segmentation import getSegmentation_fromDL
from .deepvog_torsion.torsion_lib.draw_ellipse import fit_ellipse
from .eyefitter import SingleEyeFitter
from .unprojection import reproject
from .utils import convert_vec2angle31
from .utils import save_json, load_json

"""
Ensure video has:
    1. shape (240, 320) by 
        1. resize or 
        2. crop(not yet implemented)
    2. values of float [0,1]
    3. grayscale

"""


class gaze_inferer:
    def __init__(self, model, flen, ori_video_shape, sensor_size, logger=None):
        """
        Initialize necessary parameters and load deep_learning model

        Args:
            model: Deep learning model that perform image segmentation. Pre-trained model is provided at https://github.com/pydsgz/DeepVOG/model/DeepVOG_model.py, simply by loading load_DeepVOG() with "DeepVOG_weights.h5" in the same directory. If you use your own model, it should take input of grayscale image (m, 240, 320, 1) with value float [0,1] and output (m, 240, 320, 1) with value float [0,1] where (m, 240, 320, 1) is the pupil map.
            flen (float): Focal length of camera in mm. You can look it up at the product menu of your camera
            ori_video_shape (tuple or list or np.ndarray): Original video shape from your camera, (height, width) in pixel. If you cropped the video before, use the "original" shape but not the cropped shape
            sensor_size (tuple or list or np.ndarray): Sensor size of your camera, (height, width) in mm. For 1/3 inch CMOS sensor, it should be (3.6, 4.8). Further reference can be found in https://en.wikipedia.org/wiki/Image_sensor_format and you can look up in your camera product menu
        """
        # Assertion of shape
        try:
            assert ((isinstance(flen, int) or isinstance(flen, float)))
            assert (isinstance(ori_video_shape, tuple) or isinstance(ori_video_shape, list) or isinstance(
                ori_video_shape, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size,
                                                                                                  np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size,
                                                                                                  np.ndarray))
        except AssertionError:
            print("At least one of your arguments does not have correct type")
            raise TypeError
        # logging.basicConfig(level=logging.DEBUG)
        # Parameters dealing with camera and video shape
        self.flen = flen
        self.ori_video_shape, self.sensor_size = np.array(ori_video_shape).squeeze(), np.array(sensor_size).squeeze()
        self.mm2px_scaling = np.linalg.norm(self.ori_video_shape) / np.linalg.norm(self.sensor_size)

        self.model = model
        self.logger = logger
        self.confidence_fitting_threshold = 0.96
        self.eyefitter = SingleEyeFitter(focal_length=self.flen * self.mm2px_scaling,
                                         pupil_radius=2 * self.mm2px_scaling,
                                         initial_eye_z=50 * self.mm2px_scaling)

    def fit(self, video_src, batch_size=32, print_prefix=""):
        """
        Fitting an eyeball model from video_src. After calling this method, eyeball model is stored as the attribute of the instance.
        After fitting, you can either call .save_eyeball_model() to save the model for later use, or directly call .predict() for gaze inference

        Args:
            video_src (str): Path to the video by which you want to fit an eyeball model
            batch_size (int): Batch size for each forward pass in neural network
            print_prefix (str): Printing out identifier in case you need
        """
        video_name_root, ext, vreader, (
            fitvid_m, fitvid_w, fitvid_h, fitvid_channels), shape_correct, image_scaling_factor, fps = get_video_info(
            video_src)

        # Correct eyefitter parameters in accord with the image resizing
        self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
        self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

        initial_frame, final_frame = 0, fitvid_m
        num_frames_fit = final_frame - initial_frame
        # Duration not yet implement#
        final_batch_idx = final_frame - (final_frame % batch_size)
        X_batch = np.zeros((batch_size, 240, 320, 1))
        X_batch_final = np.zeros((final_frame % batch_size, 240, 320, 1))
        for idx, frame in enumerate(vreader.nextFrame()):

            print("\r%sFitting %s (%d%%)" % (print_prefix, video_name_root + ext, (idx / fitvid_m) * 100), end="",
                  flush=True)
            frame_preprocessed = preprocess_image(frame, shape_correct)
            mini_batch_idx = idx % batch_size
            if ((mini_batch_idx != 0) and (idx < final_batch_idx)) or (idx == 0):
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed
            elif ((mini_batch_idx == 0) and (idx < final_batch_idx) or (idx == final_batch_idx)):
                Y_batch = self.model.predict(X_batch)
                self._fitting_batch(Y_batch)
                # self._write_batch(Y_batch) # Just for debugging
                X_batch = np.zeros((batch_size, 240, 320, 1))
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed
            elif ((idx > final_batch_idx) and (idx != final_frame - 1)):
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
            elif (idx == final_frame - 1):
                print("\r%sFitting %s (100%%)" % (print_prefix, video_name_root + ext), end="", flush=True)
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
                Y_batch = self.model.predict(X_batch_final)
                self._fitting_batch(Y_batch)
                # self._write_batch(Y_batch) # Just for debugging
        logging.debug("\n######### FITTING STARTS ###########")
        _ = self.eyefitter.fit_projected_eye_centre(ransac=True, max_iters=100, min_distance=3 * num_frames_fit)
        radius, _ = self.eyefitter.estimate_eye_sphere()
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            logging.error("None 3D model detected, entering python debugging mode")
        else:
            logging.debug("(Model|c,mm) Projected eye centre: {}\n".format(self.eyefitter.eye_centre.squeeze()))
            logging.debug("(Model|c,mm) Eye sphere radius: {}".format(radius / self.mm2px_scaling))
        logging.debug("######### FITTING ENDS ###########\n")
        vreader.close()
        print()

    def predict(self, video_src, output_record, batch_size=32, print_prefix="", mode="gaze", output_vis_path=None):
        """
        Inferring gaze directions from video_src and write the records (.csv) to output_record.
        Eyeball model has to be initialized first, either by calling self.fit() method, or by loading it from path with self.load_eyeball_model()

        Args:
            video_src (str): Path to the video from which you want to infer the gaze direction
            output_record (str): Path to the .csv file where you want to save the result data (e.g. pupil centre coordinates and gaze estimates)
            batch_size (int): Batch size for each forward pass in neural network
            print_prefix (str): Printing out identifier in case you need
            mode(string): "gaze"--gaze estimation;
                       "torsion"--torsional tracking;
        """

        do_visualization = bool(output_vis_path)
        
        # print('File exists: %s (%s)'%(str(os.path.exists(video_src)),video_src))
        video_name_root, ext, vreader, (infervid_m, infervid_w, infervid_h, infervid_channels), shape_correct, image_scaling_factor, fps = get_video_info(video_src)
        print('Video: %s'%video_src)
        print('This video has %d frames.'%infervid_m)

        # prepare to write result in csv file(path = output_record)
        self.results_recorder = open(output_record, "w")
        if do_visualization:
            inputdict={'-r': str(fps)}
            outputdict={'-vcodec': 'libx264',
                        '-pix_fmt': 'yuv420p',
                        '-r': str(fps), # set output framerate identical to input video
                        '-crf': '15' # set output quality (crf = Constant Rate Factor) high enough to still see iris patterns (empirically chosen)
                        }
            self.vwriter = skv.FFmpegWriter(output_vis_path,
                                            inputdict=inputdict,
                                            outputdict=outputdict)
        
        # initialize csv writers for gaze or torsion estimation
        if (mode == "gaze"):
            # Check if the eyeball model is imported
            self._check_eyeball_model_exists()

            # Correct eyefitter parameters in accord with the image resizing
            self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
            self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

            self.results_recorder.write("frame,pupil2D_x,pupil2D_y,gaze_x,gaze_y,confidence,consistence\n")

        elif (mode == "torsion"):
            self.results_recorder.write("frame, rotation\n")
            self.rotation_results = []
            self.current_iris_rotation = 0.0
            self.torsion_template_available = False
            self.torsion_template_offset = 0.0
            self.torsion_template_frame_id_global = 0
            self.torsion_template_polar_pattern = None
            self.torsion_template_polar_pattern_longer = None
            self.torsion_template_r = [-180.0,180.0]
            self.torsion_template_theta  = 90.0
            self.torsion_template_extra_radian = 0.0
            if do_visualization:
                self.time_display = 150  # range of frame when plotting graph

        else:
            raise Exception("Unknown mode {}".format(mode))
        
        
        frames_loaded_counter = 0
        frame_id_global = -1
        while frames_loaded_counter < infervid_m:
            
            # quick debugging 
            # uncomment to run full video
            #if frames_loaded_counter > 500:
            #    break
            
            # load frames into memory
            X_batch = np.zeros((batch_size, 240, 320, 1))
            counter = 0
            for idx, frame in enumerate(vreader.nextFrame()):
                X_batch[counter, :, :, :] = preprocess_image(frame, shape_correct)
                frames_loaded_counter += 1
                counter += 1
                if counter == batch_size:
                    break
            # if we reached the end of the video and there weren't enough frame to fill the batch
            if counter < batch_size:
                # reduce the size of this batch
                X_batch = X_batch[0:counter, :, :, :]
    
            # predict batch-wise on GPU
            while True:
                try:
                    Y_batch = self.model.predict(X_batch, batch_size=batch_size)
                    break
                # catching batch_size that exceeds the GPU VRAM
                except ResourceExhaustedError:
                    if batch_size > 1:
                        print("Batch size {} is too large! (causing OOM error)".format(batch_size))
                        batch_size = int(batch_size/2)
                        print("Batch size is reduced to {}".format(batch_size))
                    else:
                        raise Exception("Minimal batch size is too large.")
                # catching batch_size that exceeds the system RAM (unlikely since we now work on )
                except MemoryError:
                    print("The video or batch size is too large.")
                    print("Inference and/or pre-allocation of the results array exceeds installed RAM memory.")
                    break
            
            # perform postprocessing
            if (mode == "gaze"):
                # put former "self._infer_batch" inline
                
                if do_visualization:
                    vid_frames = np.around(X_batch * 255).astype(np.int)
                    vid_frame_shape_2d = (vid_frames.shape[1], vid_frames.shape[2])
                    
        
                for frame_id, (X_i, Y_i) in enumerate(zip(X_batch, Y_batch)):
                    frame_id_global = frames_loaded_counter - batch_size + frame_id
                    #print("Processing {}%".format(int((frame_id+1)*100. / len(X_batch))),end="\r")
                    pred = Y_i[:, :, 0]
                    _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pred)
                    (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
                    
                    if do_visualization:
                        vid_frame = vid_frames[frame_id]
                        
                    if centre is not None:
                        p_list, n_list, _, consistence = self.eyefitter.gen_consistent_pupil()
                        p1, n1 = p_list[0], n_list[0]
                        px, py, pz = p1[0, 0], p1[1, 0], p1[2, 0]
                        x, y = convert_vec2angle31(n1)
                        positions = (px, py, pz, centre[0], centre[1])  # Pupil 3D positions and 2D projected positions
                        gaze_angles = (x, y)  # horizontal and vertical gaze angles
                        inference_confidence = (ellipse_confidence, consistence)
                        self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame_id_global + 1, centre[0], centre[1],
                                                                                x, y,
                                                                                ellipse_confidence, consistence))
                        if do_visualization:
                            # # Code below is for drawing video
                            ellipse_centre_np = np.array(centre)
                            projected_eye_centre = reproject(self.eyefitter.eye_centre,
                                                             self.eyefitter.focal_length)  # shape (2,1)
                            # The line below is for translation from camera coordinate system (centred at image centre)
                            # to numpy's indexing frame. You substrate the vector by the half of the video's 2D shape. Col = x-axis,
                            # Row = y-axis
                            projected_eye_centre += np.array(vid_frame_shape_2d).T.reshape(-1, 1) / 2
                            
                            vid_frame = draw_vis_on_frame(vid_frame, vid_frame_shape_2d, ellipse_info,
                                                          ellipse_centre_np,
                                                          projected_eye_centre, gaze_vec=n1)
                            self.vwriter.writeFrame(vid_frame)
                    else:
                        positions, gaze_angles, inference_confidence = None, None, None
                        self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame_id_global + 1, np.nan, np.nan,
                                                                                np.nan, np.nan,
                                                                                np.nan, np.nan))
                        if do_visualization:
                            vid_frame = np.stack((vid_frame[:, :, 0],) * 3, axis=-1)
                            self.vwriter.writeFrame(vid_frame)
                            
                print("\rProcessing (%0.2f%%, frame %d of %d)."%(100.0*frame_id_global/infervid_m,frame_id_global,infervid_m),end='\r')
                
            # put former "self._infer_torsion_batch" inline
            elif (mode == 'torsion'):
                
                for frame_id, (X_i, Y_i) in enumerate(zip(X_batch, Y_batch)):
                    frame_id_global += 1 
                    
                    # binarize all segmentations (pupil, iris, glints, eye, background)
                    pred_masked = np.ma.masked_where(Y_i < 0.5, Y_i)
                    
                    # Initialize frames and maps: int->float, rgb->gray
                    frame = img_as_float(X_i)  # frame ~ (240, 320, 1)
                    frame_gray = rgb2gray(frame)[0, :, :, 0]  # frame_gray ~ (240, 320)
                    
                    frame_rgb = np.zeros((frame.shape[0], frame.shape[1], 3))  # frame_rgb ~ (240, 320, 3)
                    frame_rgb[:, :, :] = frame_gray.reshape(frame_gray.shape[0], frame_gray.shape[1], 1)
                    
                    # extract iris and pupil regions, fit pupil ellipse
                    useful_map, (pupil_map, _, _, _) = getSegmentation_fromDL(Y_i)
                    _, (pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked) = getSegmentation_fromDL(
                        pred_masked)
                    rr, _, centre, _, _, _, _, _ = fit_ellipse(pupil_map, 0.5)
                    
                    if self.torsion_template_available: #'polar_pattern_template_longer' is not None: # ref_frame > -1:
                        # we have already a valid template
                        # if pupil detection failed, assume torsion as nan, otherwise compute torsion
                        if centre is None:
                            # pupil detection failed
                            # skip this frame as reference
                            rotation = np.nan
                            rotated_info = None
                            self.current_iris_rotation = rotation
                        else:
                            # pupil detected --> compute torsion as well
                            rotation, rotated_info, corr, quality_measures = findTorsion(self.torsion_template_polar_pattern_longer, 
                                                                    frame_gray, 
                                                                    useful_map,
                                                                    center=centre, 
                                                                    filter_sigma=100, 
                                                                    adhist_times=2)
                            #print('Template rotation before template update: %0.2f'%self.current_iris_rotation)
                            self.current_iris_rotation = self.torsion_template_offset + rotation
                            
                            # updating the template necessary?
                            if ( (quality_measures['polar_coverage_percent'] > self.torsion_template_quality_measures['polar_coverage_percent']) & (quality_measures['polar_xgradient_sum'] > self.torsion_template_quality_measures['polar_xgradient_sum']) ):
                                self.updateTorsionTemplate(frame_id_global, frame_gray, useful_map, centre)
                    else:
                        # there is no template yet (beginning of video)
                        # we need to check whether this is a valid template (pupil detection successful?)
                        if centre is None:
                            # no pupil detected --> can't be used as template either
                            # skip this frame as reference, wait for first frame in video with a detectable pupil
                            rotation = np.nan
                            rotated_info = None
                            self.current_iris_rotation = rotation
                        else:
                            # we have found a valid reference frame
                            #ref_frame = frame_id
                            # store global index of this image/frame in the video
                            self.updateTorsionTemplate(frame_id_global, frame_gray, useful_map, centre)
                            rotation = self.current_iris_rotation
                            rotated_info = self.torsion_template_rotated_info
                            extra_radian = self.torsion_template_extra_radian
                            
                    self.rotation_results.append(self.current_iris_rotation)
                    self.results_recorder.write("{},{}\n".format(frame_id_global, self.current_iris_rotation))
                    
                    if do_visualization:
                        # Drawing the frames of visualisation video
                        # render the torsional rotation curve (top-right subplot)
                        rotation_plot_arr = plot_rotation_curve(frame_id_global, self.time_display, self.rotation_results)
                        # render the segmentation result (bottom-left subplot)
                        segmented_frame = draw_segmented_area(frame_gray, pupil_map_masked, iris_map_masked,
                                                              glints_map_masked, visible_map_masked)
                        # render the polar transformation of template vs rotated iris patterns (bottom-right)
                        polar_transformed_graph_arr = plot_polar_transformed_graph(
                                                            (self.torsion_template_polar_pattern, 
                                                             self.torsion_template_r, 
                                                             self.torsion_template_theta), 
                                                            rotated_info, 
                                                            extra_radian)
                        frames_to_draw = (frame_rgb, rotation_plot_arr, segmented_frame, polar_transformed_graph_arr)
                        final_output = build_final_output_frame(frames_to_draw)
                        self.vwriter.writeFrame(final_output)
                        
                    print("\rProcessing (%0.2f%%, frame %d of %d)."%(100.0*frame_id_global/infervid_m,frame_id_global,infervid_m),end='\r')
            else:
                # dead code
                pass
            # end: if mode==gaze/torsion/else
        # end: while frame_counter < infervid_m:
        print("Processing done. Writing results.")
        self.results_recorder.close()
        if output_vis_path:
            self.vwriter.close()
    
    def updateTorsionTemplate(self, frame_id_global, frame_gray, useful_map, centre, strategy='replace'):
        # generate new template from current gray image 
        polar_pattern_template, \
            polar_pattern_template_longer, \
            r_template, \
            theta_template, \
            extra_radian, \
            quality_measures = genPolar(frame_gray, useful_map, center=centre, template=True,
                              filter_sigma=100, adhist_times=2, apply_gaussian_noise=True)
        
        if strategy == 'replace':
            #print('Shape of polar_pattern_template: %s'%str(polar_pattern_template.shape))
            #print(quality_measures)
            #print('Template rotation after: %0.2f'%self.current_iris_rotation)
            rotated_info = (polar_pattern_template, r_template, theta_template)
            # store to internal memory
            self.torsion_template_available = True
            self.torsion_template_offset = self.current_iris_rotation # when setting a new template, we store the current accumulated torsion "as is" into offset
            self.torsion_template_frame_id_global = frame_id_global
            self.torsion_template_rotated_info = rotated_info
            self.torsion_template_polar_pattern = polar_pattern_template
            self.torsion_template_polar_pattern_longer = polar_pattern_template_longer
            self.torsion_template_extra_radian = extra_radian
            self.torsion_template_quality_measures = quality_measures
            self.torsion_template_r = r_template
            self.torsion_template_theta  = theta_template
        else:
            print('Template update strategy "%s" not implemented.')
            return

    
    def save_eyeball_model(self, path):
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            print("3D eyeball model not found")
            raise Exception("3D eyeball model not found")
        else:
            save_dict = {"eye_centre": self.eyefitter.eye_centre.tolist(),
                         "aver_eye_radius": self.eyefitter.aver_eye_radius}
            save_json(path, save_dict)
            
    def load_eyeball_model(self, path):
        loaded_dict = load_json(path)
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            self.eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
            self.eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]
            
        else:
            logging.warning("3D eyeball exists and reloaded")
            
    def _fitting_batch(self, Y_batch):
        for Y_each in Y_batch:
            pred_each = Y_each[:, :, 0]
            _, _, _, _, (_, _, centre, w, h, radian, ellipse_confidence) = self.eyefitter.unproject_single_observation(
                pred_each)
            
            if (ellipse_confidence > self.confidence_fitting_threshold) and (centre is not None):
                self.eyefitter.add_to_fitting()
                
    def _infer_batch(self, Y_batch, idx):
        for batch_idx, Y_each in enumerate(Y_batch):
            frame = idx + batch_idx + 1
            pred_each = Y_each[:, :, 0]
            _, _, _, _, (_, _, centre, w, h, radian, ellipse_confidence) = self.eyefitter.unproject_single_observation(
                pred_each)

            if centre is not None:
                p_list, n_list, _, consistence = self.eyefitter.gen_consistent_pupil()
                p1, n1 = p_list[0], n_list[0]
                px, py, pz = p1[0, 0], p1[1, 0], p1[2, 0]
                x, y = convert_vec2angle31(n1)
                positions = (px, py, pz, centre[0], centre[1])  # Pupil 3D positions and 2D projected positions
                gaze_angles = (x, y)  # horizontal and vertical gaze angles
                inference_confidence = (ellipse_confidence, consistence)
                self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame, centre[0], centre[1],
                                                                        x, y,
                                                                        ellipse_confidence, consistence))

            else:
                positions, gaze_angles, inference_confidence = None, None, None
                self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame, np.nan, np.nan,
                                                                        np.nan, np.nan,
                                                                        np.nan, np.nan))
        return positions, gaze_angles, inference_confidence

    def _check_eyeball_model_exists(self):
        try:
            assert isinstance(self.eyefitter.eye_centre, np.ndarray)
            assert self.eyefitter.eye_centre.shape == (3, 1)
            assert self.eyefitter.aver_eye_radius is not None
        except AssertionError as e:
            logging.error("You must initialize 3D eyeball parameters first by fit() function")
            raise e
    
    def _update_template(self):
        # e.g. based on template image contrast/sharpness
        # for a discussion of (basic) methods, see:
        # https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
        
        # other ideas: continuously reconstruct an optimally sharp iris image (with maximally little occlusion)
        # e.g. based on LoG-pyramids and HDR merging
        
        # current implementation: based on average x-gradient
        pass


def get_video_info(video_src):
    video_name_with_ext = os.path.split(video_src)[1]
    video_name_root, ext = os.path.splitext(video_name_with_ext)
    vreader = skv.FFmpegReader(video_src)
    m, w, h, channels = vreader.getShape()
    fps = vreader.inputfps
    image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))
    shape_correct = inspectVideoShape(w, h)
    return video_name_root, ext, vreader, (m, w, h, channels), shape_correct, image_scaling_factor, fps


def inspectVideoShape(w, h):
    if (w, h) == (240, 320):
        return True
    else:
        return False


def computeCroppedShape(ori_video_shape, crop_size):
    video = np.zeros(ori_video_shape)
    cropped = video[crop_size[0]:crop_size[1], crop_size[2], crop_size[3]]
    return cropped.shape


def preprocess_image(img, resizing):
    output_img = np.zeros((240, 320, 1))
    img = img / 255
    img = rgb2gray(img)[0,:,:,0]
    if resizing == True:
        img = resize(img, (240, 320))
    output_img[:, :, :] = img.reshape(240, 320, 1)
    return output_img


def plot_rotation_curve(idx, time_display, rotation_results, y_lim=(-4, 4)):
    # fig, ax = plt.subplots( figsize=(3.2,2.4)) #width, height

    fig = Figure(figsize=(3.2, 2.4),dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()

    if idx < time_display:
        ax.plot(np.arange(0, idx), rotation_results[0:idx], color="b", label="DeepVOG 3D")
        ax.set_xlim(0, time_display)
    else:
        ax.plot(np.arange(idx - time_display, idx), rotation_results[idx - time_display:idx],
                color="b", label="DeepVOG 3D")
        ax.set_xlim(idx - time_display, idx)
    ax.legend()
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_yticks(np.arange(y_lim[0], y_lim[1]))
    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = (np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8) / 255).reshape(h, w, 3)

    return buf


def draw_segmented_area(frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked,
                        visible_map_masked):
    # Plot segmented area
    # fig, ax = plt.subplots(figsize=(3.2,2.4))
    fig = Figure(figsize=(3.2, 2.4),dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.imshow(frame_gray, vmax=1, vmin=0, cmap="gray")
    ax.imshow(visible_map_masked, cmap="autumn", vmax=1, vmin=0, alpha=0.2)
    ax.imshow(iris_map_masked, cmap="GnBu", vmax=1, vmin=0, alpha=0.2)
    ax.imshow(pupil_map_masked, cmap="hot", vmax=1, vmin=0, alpha=0.2)
    ax.imshow(glints_map_masked, cmap="OrRd", vmax=1, vmin=0, alpha=0.2)
    ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = (np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8) / 255).reshape(h, w, 3)
    return buf


def plot_polar_transformed_graph(template_info, rotated_info, extra_radian):
    (polar_pattern, r, theta) = template_info
    if rotated_info is not None:
        (polar_pattern_rotated, r_rotated, theta_rotated) = rotated_info
    else:
        polar_pattern_rotated, r_rotated, theta_rotated = np.zeros(polar_pattern.shape), r, theta

    # x axis correction
    theta_longer = np.rad2deg(theta) - np.rad2deg((theta.max() - theta.min()) / 2)
    theta_shorter = np.rad2deg(theta_rotated) - np.rad2deg((theta_rotated.max() - theta_rotated.min()) / 2)
    theta_extra = np.rad2deg(extra_radian)

    # Plotting
    # fig, ax = plt.subplots(2, figsize=(3.2,2.4))
    fig = Figure(figsize=(3.2, 2.4),dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.subplots(2)

    ax[0].imshow(polar_pattern, cmap="gray", aspect='auto', vmin=0.0, vmax=1.0,
                 extent=(theta_shorter.min(), theta_shorter.max(), r.max(), r.min()),
                 )
    ax[0].set_title("Template")
    ax[0].yaxis.set_visible(False)
    ax[1].imshow(polar_pattern_rotated, cmap="gray", aspect='auto', vmin=0.0, vmax=1.0,
                 extent=(theta_shorter.min(), theta_shorter.max(), r_rotated.max(), r_rotated.min())
                 )
    ax[1].set_title("Rotated pattern")
    ax[1].yaxis.set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = (np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8) / 255).reshape(h, w, 3)
    return buf


def build_final_output_frame(frames_to_draw):
    """
    args:
        frames_to_draw: tuple with length 4. Starting from top left corner in clockwise direction.
    """
    height, width = 240, 320
    final_output = np.zeros((height * 2, width * 2, 3))
    final_output[0:height, 0:width, :] = frames_to_draw[0]
    final_output[0:height, width:width * 2, :] = frames_to_draw[1]
    final_output[height:height * 2, 0:width, :] = frames_to_draw[2]
    final_output[height:height * 2, width:width * 2, :] = frames_to_draw[3]
    final_output = (final_output * 255).astype(np.uint8)
    return final_output


def draw_vis_on_frame(origin_vid_frame, vid_frame_shape_2d, ellipse_info, ellipse_centre_np, projected_eye_centre,
                      gaze_vec):
    vid_frame = np.stack((origin_vid_frame[:, :, 0],) * 3, axis=-1)

    # Draw pupil ellipse
    vid_frame = draw_ellipse(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                             ellipse_info=ellipse_info, color=[255, 255, 0])

    # Draw from eyeball centre to ellipse centre (just connecting two points)
    vec_with_length = ellipse_centre_np - projected_eye_centre.squeeze()
    vid_frame = draw_line(output_frame=vid_frame, frame_shape=vid_frame_shape_2d, o=projected_eye_centre,
                          l=vec_with_length, color=[0, 0, 255])

    # Draw gaze vector originated from ellipse centre
    vid_frame = draw_line(output_frame=vid_frame, frame_shape=vid_frame_shape_2d, o=ellipse_centre_np,
                          l=gaze_vec * 50, color=[255, 0, 0])

    # Draw small circle at the ellipse centre
    vid_frame = draw_circle(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                            centre=ellipse_centre_np, radius=5, color=[0, 255, 0])
    return vid_frame


def draw_line(output_frame, frame_shape, o, l, color=[255, 0, 0]):
    """

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    o : list or tuple or numpy.darray
        Origin of the line, with shape (2,) denoting (x, y).
    l : list or tuple or numpy.darray
        Vector with length. Body of the line. Shape = (2, ), denoting (x, y)
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the ellipse drawn.
    """
    R, G, B = color
    rr, cc = line(int(np.round(o[0])), int(np.round(o[1])), int(np.round(o[0] + l[0])), int(np.round(o[1] + l[1])))
    rr[rr > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc[cc > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc, rr, 1] = G
    output_frame[cc, rr, 2] = B
    return output_frame


def draw_ellipse(output_frame, frame_shape, ellipse_info, color=[255, 255, 0]):
    """
    Draw a circle on an image or video frame. Drawing will be discretized.

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    ellipse_info : list or tuple
        Information of ellipse parameters. (rr, cc, centre, w, h, radian, ellipse_confidence).
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the ellipse drawn.
    """

    R, G, B = color
    (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
    rr[rr > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc[cc > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc, rr, 1] = G
    output_frame[cc, rr, 2] = B
    return output_frame


def draw_circle(output_frame, frame_shape, centre, radius, color=[255, 0, 0]):
    """
    Draw a circle on an image or video frame. Drawing will be discretized.

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    centre : list or tuple or numpy.darray
        x,y coordinate of the circle centre
    radius : int or float
        Radius of the circle to draw.
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the circle drawn.
    """

    R, G, B = color
    rr_p1, cc_p1 = circle_perimeter(int(np.round(centre[0])), int(np.round(centre[1])), radius)
    rr_p1[rr_p1 > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc_p1[cc_p1 > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr_p1[rr_p1 < 0] = 0
    cc_p1[cc_p1 < 0] = 0
    output_frame[cc_p1, rr_p1, 0] = R
    output_frame[cc_p1, rr_p1, 1] = G
    output_frame[cc_p1, rr_p1, 2] = B
    return output_frame
