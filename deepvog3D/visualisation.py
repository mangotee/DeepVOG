from .inferer import gaze_inferer
from skimage.draw import ellipse_perimeter, line, circle_perimeter, line_aa
from .eyefitter import SingleEyeFitter
from .unprojection import reproject
from .utils import convert_vec2angle31
import skvideo.io as skv
import numpy as np
import logging
import matplotlib.pyplot as plt
from skimage import img_as_float
from skvideo.utils import rgb2gray
from .deepvog_torsion.torsion_lib.Segmentation import getSegmentation_fromDL
from .deepvog_torsion.torsion_lib.draw_ellipse import fit_ellipse
from .deepvog_torsion.torsion_lib.CrossCorrelation import genPolar, findTorsion

def draw_line(output_frame, frame_shape, o, l, color = [255,0,0]):
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
    rr, cc = line(int(np.round(o[0])), int(np.round(o[1])), int(np.round(o[0]+l[0])), int(np.round(o[1]+l[1])))
    rr[rr>int(frame_shape[1])-1] = frame_shape[1]-1
    cc[cc>int(frame_shape[0])-1] = frame_shape[0]-1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc,rr,1] = G
    output_frame[cc,rr,2] = B
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


class Visualizer(gaze_inferer):
    """
    This Class is not intended for run-time deployment. It is for demonstration of how to visualize the gaze vector and
    fitted ellipse. Modification is needed to integrate into the framework
    """

    def predict(self, video_src, output_record, batch_size=32, print_prefix="", mode = "gaze", output_vis_path=None):
        """
        Overriden from gaze_inferer.predict()
        Added argument : output_vis_path

        It is 99% same with gaze_inferer.predict() method, except with added _infer_vis_batch() method.
        Visualizer._infer_vis_batch() is intended to infer gaze as well as drawing visualisation video output


        Parameters
        ----------
        video_src : str
            input video
        output_record : str
            record .csv
        batch_size : int
            Inference batch size
        print_prefix : str
            Printing identifier
        mode: str
            Determine to do gaze estimation or torsional tracking
        output_vis_path : str
            The path of the output visualization that will be drawn

        Returns
        -------
        positions, gaze_angles, inference_confidence

        """
        video_name_root, ext, vreader, (infervid_m, infervid_w, infervid_h,
                                        infervid_channels), shape_correct, image_scaling_factor = self._get_video_info(video_src)

        if (mode == "gaze"):
            # Check if the eyeball model is imported
            self._check_eyeball_model_exists()

            # Correct eyefitter parameters in accord with the image resizing
            self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
            self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor


        # Initialize path and video writer
        if output_vis_path:
            self.output_vid_path = output_vis_path
            self.vwriter = skv.FFmpegWriter(self.output_vid_path)

        # Initialize for result recording
        self.results_recorder = open(output_record, "w")
        if(mode == "gaze"):
            self.results_recorder.write("frame,pupil2D_x,pupil2D_y,gaze_x,gaze_y,confidence,consistence\n")
        elif(mode=="torsion"):
            self.results_recorder.write("frame, rotation\n")
            self.rotation_results = []
            self.time_display = 150 # range of frame when plotting graph
           

        final_batch_size = infervid_m % batch_size
        final_batch_idx = infervid_m - final_batch_size
        X_batch = np.zeros((batch_size, 240, 320, 1))
        X_batch_final = np.zeros((infervid_m % batch_size, 240, 320, 1))
        for idx, frame in enumerate(vreader.nextFrame()):

            print("\r%sInferring %s (%d%%)" % (print_prefix, video_name_root + ext, (idx / infervid_m) * 100), end="",
                  flush=True)
            frame_preprocessed = self._preprocess_image(frame, shape_correct)
            mini_batch_idx = idx % batch_size

            # Before reaching the batch size, stack the array
            if ((mini_batch_idx != 0) and (idx < final_batch_idx)) or (idx == 0):
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

            # After reaching the batch size, but not the final batch, predict and infer angles
            elif ((mini_batch_idx == 0) and (idx < final_batch_idx) or (idx == final_batch_idx)):
                Y_batch = self.model.predict(X_batch)
                # =============== infer angles by batch here ====================
                if(mode == "gaze"):
                    if output_vis_path:
                        positions, gaze_angles, inference_confidence = self._infer_vis_batch(X_batch, Y_batch, idx - batch_size)
                    else:
                        positions, gaze_angles, inference_confidence = self._infer_batch(Y_batch, idx - batch_size)
                elif(mode == "torsion"):
                    if output_vis_path:
                        self._infer_torsion_vis_batch(X_batch, Y_batch)                        
                    else:
                        self._infer_torsion_batch(X_batch, Y_batch)
                X_batch = np.zeros((batch_size, 240, 320, 1))
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

            # Within the final batch but not yet reaching the last index, stack the array
            elif ((idx > final_batch_idx) and (idx != infervid_m - 1)):
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed

            # Within the final batch and reaching the last index, predict and infer angles
            elif (idx == infervid_m - 1):
                print("\r%sInferring %s (100%%)" % (print_prefix, video_name_root + ext), end="", flush=True)
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
                Y_batch = self.model.predict(X_batch_final)
                # =============== infer angles by batch here ====================
                if(mode == "gaze"):
                    if output_vis_path:
                        positions, gaze_angles, inference_confidence = self._infer_vis_batch(X_batch, Y_batch, idx - final_batch_size)
                    else:
                        positions, gaze_angles, inference_confidence = self._infer_batch(Y_batch, idx - final_batch_size)
                elif(mode == "torsion"):
                    if output_vis_path:
                        self._infer_torsion_vis_batch(X_batch_final, Y_batch)
                    else:
                        self._infer_torsion_batch(X_batch_final, Y_batch)
            else:
                import pdb
                pdb.set_trace()
        self.results_recorder.close()
        if output_vis_path:
            self.vwriter.close()
        print()

    def _infer_vis_batch(self, X_batch, Y_batch, idx):

        # Convert video frames to 8 bit integer format
        video_frames_batch = np.around(X_batch * 255).astype(int)
        vid_frame_shape_2d = (video_frames_batch.shape[1], video_frames_batch.shape[2])

        for batch_idx, (X_each, Y_each) in enumerate(zip(X_batch, Y_batch)):
            frame = idx + batch_idx + 1
            pred_each = Y_each[:, :, 0]
            # In the method of eyefitter.unproject_single_observation(), you will need to call fit_ellipse() instead of
            # fit_ellipse_compact() to get the indexes for drawing (rr and cc). I have this modification here.
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pred_each)
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
            vid_frame = video_frames_batch[batch_idx,]
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
                # # Code below is for drawing video
                ellipse_centre_np = np.array(centre)
                projected_eye_centre = reproject(self.eyefitter.eye_centre, self.eyefitter.focal_length) # shape (2,1)
                # The line below is for translation from camera coordinate system (centred at image centre)
                # to numpy's indexing frame. You substrate the vector by the half of the video's 2D shape. Col = x-axis,
                # Row = y-axis
                projected_eye_centre += np.array(vid_frame_shape_2d).T.reshape(-1, 1)/2

                vid_frame = self._draw_vis_on_frame(vid_frame, vid_frame_shape_2d, ellipse_info, ellipse_centre_np,
                                                    projected_eye_centre, gaze_vec=n1)
                self.vwriter.writeFrame(vid_frame)

            else:
                positions, gaze_angles, inference_confidence = None, None, None
                self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame, np.nan, np.nan,
                                                                        np.nan, np.nan,
                                                                        np.nan, np.nan))
                vid_frame = np.stack((vid_frame[:,:,0],)*3, axis=-1)
                self.vwriter.writeFrame(vid_frame)
        return positions, gaze_angles, inference_confidence

    def _infer_torsion_vis_batch (self, X_batch, Y_batch, update_template = False):
        # do visulisation for torsion

        for idx, pred in enumerate(Y_batch):

            pred_masked = np.ma.masked_where(pred < 0.5, pred)

            # Initialize frames and maps
            frame = img_as_float(X_batch[idx]) # frame ~ (240, 320, 1)
            frame_gray = rgb2gray(frame)[0,:,:,0] # frame_gray ~ (240, 320)
            frame_rgb = np.zeros((frame.shape[0], frame.shape[1], 3)) # frame_rgb ~ (240, 320, 3)
            frame_rgb[:,:,:] = frame_gray.reshape(frame_gray.shape[0], frame_gray.shape[1], 1)
            useful_map, (pupil_map, _, _, _) = getSegmentation_fromDL(pred)
            _, (pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked) = getSegmentation_fromDL(pred_masked)
            rr, _, centre, _, _, _, _, _ = fit_ellipse(pupil_map, 0.5)
                
            if centre == None:
                continue

            # Cross-correlation
            if idx == 1 :
                try:
                    polar_pattern_template, polar_pattern_template_longer, r_template, theta_template, extra_radian = genPolar(frame_gray, useful_map, center = centre, template = True,
                                                                                        filter_sigma = 100, adhist_times = 2)
                    rotated_info = (polar_pattern_template, r_template, theta_template)
                    rotation = 0                                                                        
                except:
                    import pdb
                    pdb.set_trace()
            elif rr is not None:
                # for finding the rotation value and determine if it is needed to update
                rotation, rotated_info , _  = findTorsion(polar_pattern_template_longer, frame_gray, useful_map, center = centre,
                                                        filter_sigma = 100, adhist_times = 2)


                if (update_template == True) and rotation == 0:
                    polar_pattern_template, polar_pattern_template_longer, r_template, theta_template, extra_radian = genPolar(frame_gray, useful_map, center = centre, template = True,
                                                                                filter_sigma = 100, adhist_times = 2)
            else:
                rotation, rotated_info = np.nan, None

            
            self.rotation_results.append(rotation)
            self.results_recorder.write("{},{}\n".format(idx, rotation))
            
            # Drawing the frames of visualisation video
            rotation_plot_arr = self._plot_rotation_curve(idx)
            segmented_frame = self._draw_segmented_area( frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked)
            polar_transformed_graph_arr = self._plot_polar_transformed_graph((polar_pattern_template, r_template, theta_template), rotated_info, extra_radian)
            frames_to_draw = (frame_rgb, rotation_plot_arr, segmented_frame, polar_transformed_graph_arr)
            final_output = self._build_final_output_frame(frames_to_draw)
            self.vwriter.writeFrame(final_output)
            

    def _plot_rotation_curve(self, idx, y_lim = (-4, 4)):
        fig, ax = plt.subplots( figsize=(3.2,2.4)) #width, height 
        
        try:
            if idx < self.time_display:    
                ax.plot(np.arange(0, idx), self.rotation_results[0:idx], color = "b", label = "DeepVOG 3D")
                ax.set_xlim(0,self.time_display)
            else:
                ax.plot(np.arange(idx- self.time_display, idx), self.rotation_results[idx-self.time_display:idx], color = "b", label = "DeepVOG 3D")
                ax.set_xlim(idx-self.time_display, idx)
            ax.legend()
            ax.set_ylim(y_lim[0],y_lim[1])
            ax.set_yticks(np.arange(y_lim[0],y_lim[1]))
            plt.tight_layout()
        except:
            pass  
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf
            
    def _draw_segmented_area(self, frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked):
        # Plot segmented area
        fig, ax = plt.subplots(figsize=(3.2,2.4))
        ax.imshow(frame_gray, vmax=1, vmin=0, cmap="gray")
        ax.imshow(visible_map_masked, cmap="autumn", vmax=1, vmin=0, alpha = 0.2)
        ax.imshow(iris_map_masked, cmap="GnBu", vmax=1, vmin=0, alpha = 0.2)
        ax.imshow(pupil_map_masked, cmap="hot", vmax=1, vmin=0, alpha = 0.2)
        ax.imshow(glints_map_masked, cmap="OrRd", vmax=1, vmin=0, alpha = 0.2)
        ax.set_axis_off()
        plt.tight_layout()
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf

    def _plot_polar_transformed_graph(self, template_info, rotated_info, extra_radian ):
        (polar_pattern, r, theta) = template_info
        if rotated_info is not None:
            (polar_pattern_rotated, r_rotated, theta_rotated) = rotated_info
        else:
            polar_pattern_rotated, r_rotated, theta_rotated = np.zeros(polar_pattern.shape), r, theta

        # x axis correction
        theta_longer = np.rad2deg(theta) - np.rad2deg((theta.max()-theta.min() )/2)
        theta_shorter = np.rad2deg(theta_rotated) - np.rad2deg((theta_rotated.max() - theta_rotated.min())/2)
        theta_extra = np.rad2deg(extra_radian)
        
        # Plotting
        fig, ax = plt.subplots(2, figsize=(3.2,2.4))
        ax[0].imshow(polar_pattern, cmap="gray", extent=(theta_shorter.min(), theta_shorter.max(), r.max(), r.min()), aspect='auto')
        ax[0].set_title("Template")
        ax[1].imshow(polar_pattern_rotated, cmap="gray", extent=(theta_shorter.min(), theta_shorter.max(), r_rotated.max(), r_rotated.min()), aspect='auto')
        ax[1].set_title("Rotated pattern")
        plt.tight_layout()
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf

    def _build_final_output_frame(self, frames_to_draw):
        """
        args:
            frames_to_draw: tuple with length 4. Starting from top left corner in clockwise direction.
        """
        height, width = 240, 320
        final_output = np.zeros((height*2, width*2, 3))
        final_output[0:height, 0:width, : ] = frames_to_draw[0]
        final_output[0:height, width:width*2, :] = frames_to_draw[1]
        final_output[height:height*2, 0:width, :] = frames_to_draw[2]
        final_output[height:height*2, width:width*2, :] = frames_to_draw[3]
        final_output = (final_output*255).astype(np.uint8)
        return final_output

    @staticmethod
    def _draw_vis_on_frame(origin_vid_frame, vid_frame_shape_2d, ellipse_info, ellipse_centre_np, projected_eye_centre,
                           gaze_vec):

        vid_frame = np.stack((origin_vid_frame[:,:,0],)*3, axis=-1)

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
