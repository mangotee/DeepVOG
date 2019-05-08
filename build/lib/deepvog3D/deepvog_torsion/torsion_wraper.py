from .torsion import offline_inferer
import os

def torsional_inference(video_dir, pred_dir, output_dir):
    '''
    Args:
        video_dir: directory containing videos to infer torsion
        pred_dir: directory containing output from DeepVOG3D model predictions (.npy file)
        output_dir: directory to save torsional results
    '''
    
    for path in os.listdir(video_dir):
        name = os.path.splitext(path)[0]
        video_path = os.path.join(video_dir, path)
        pred_path = os.path.join(pred_dir, name+".npy")
        output_record_path = os.path.join(output_dir,"output_records",name+".csv")
        output_video_path = os.path.join(output_dir,"output_visualisation",name+".mp4")

        torsioner = offline_inferer(video_path, pred_path)
        torsioner.plot_video(output_video_path, output_record_path, update_template = False)
