"""
  MODEL             SKLearn model file created using training.py
  INPUT             Input video file
  OUTPUT            Output video filename
"""

from detection_functions import *
from training_functions import *
from tracking_functions import *
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

def process_image(image, params):
    config, clf = params['clf_config'], params['clf']

    if 'windows' not in params:
        pyramid = [((64, 64),  [400, 500]),
                   ((96, 96),  [400, 500]),
                   ((128, 128),[450, 600]),
                  ]

        image_size = image.shape[:2]
        params['windows'] = create_windows(pyramid, image_size)

    all_windows = params['windows']
    if params['cache_enabled']:
        cache = process_image.cache
        if cache['heatmaps'] is None:
            cache['heatmaps'] = collections.deque(maxlen=params['heatmap_cache_length'])
            cache['last_heatmap'] = np.zeros(image.shape[:2])
        if 'tracker' not in cache:
            cache['tracker'] = VehicleTracker(image.shape)
        frame_ctr = cache['frame_ctr']
        tracker = cache['tracker']
        cache['frame_ctr'] += 1

    windows = itertools.chain(*all_windows)

    measurements = multiscale_detect(image, clf, config, windows)
    current_heatmap = update_heatmap(measurements, image.shape)
    if not params['cache_enabled']:
        thresh_heatmap = current_heatmap
        thresh_heatmap[thresh_heatmap < params['heatmap_threshold']] = 0
        cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

        labels = label(thresh_heatmap)
        im2 = draw_labeled_bboxes(np.copy(image), labels)
    else:
        cache['heatmaps'].append(current_heatmap)
        thresh_heatmap = sum(cache['heatmaps'])

        thresh_heatmap[thresh_heatmap < params['heatmap_threshold']] = 0
        cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)
        labels = label(thresh_heatmap)
        Z = []
        for car_number in range(1, labels[1]+1):
            nonzeroy, nonzerox = np.where(labels[0] == car_number)
            Z.append((np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)))
        tracker.track(Z)
        im2 = tracker.draw_bboxes(np.copy(image))

    return im2

def clear_cache():
    process_image.cache = {
        'last_heatmap': None,
        'heatmaps': None,
        'frame_ctr': 0
    }

#from docopt import docopt

if __name__ == '__main__':
    #arguments = docopt(__doc__, version='Vehicle Detection v1.0')

    clear_cache()
    model_file = 'models/clf.pkl'
    in_file = 'input/project_video.mp4'
    out_file = 'output/project_video_output.mp4'

    print('Loading model ...')
    data = joblib.load(model_file)

    clf = data['model']
    config = data['config']

    clear_cache()
    params = {}
    params['clf_config'] = config
    params['clf'] = clf
    params['cache_enabled'] = True
    params['heatmap_cache_length'] = 25
    params['heatmap_threshold'] = 10

    print('Processing video ...')
    clip2 = VideoFileClip(in_file)
    vid_clip = clip2.fl_image(lambda i: process_image(i, params))
    vid_clip.write_videofile(out_file, audio=False)
