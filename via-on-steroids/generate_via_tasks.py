import os
import glob
import json


def new_data():
    data = {
        '_via_settings': {
            'ui': {
                'annotation_editor_height': 25,
                'annotation_editor_fontsize': 0.8,
                'leftsidebar_width': 18,
                'image_grid': {
                    'img_height': 80,
                    'rshape_fill': 'none',
                    'rshape_fill_opacity': 0.3,
                    'rshape_stroke': 'yellow',
                    'rshape_stroke_width': 2,
                    'show_region_shape': True,
                    'show_image_policy': 'all'
                },
                'image': {
                    'region_label': '__via_region_id__',
                    'region_color': '__via_default_region_color__',
                    'region_label_font': '10px Sans',
                    'on_image_annotation_editor_placement': 'NEAR_REGION'
                }
            },
            'core': {
                'buffer_size': '18',
                'filepath': {},
                'default_filepath': 'test/path/thing/'  # CHANGE THIS
            },
            'project': {'name': 'CHANGE THIS'}
        },
        '_via_img_metadata': {},  # POPULATE THIS
        '_via_attributes': {'region': {}, 'file': {}}
    }
    return data


if __name__ == '__main__':
    data_root = input(
        'Please enter the data root path INCLUDING THE FINAL "/" :')
    embryo_paths = glob.glob(os.path.join(data_root, '*'))
    for embryo_path in embryo_paths:
        embryo = os.path.basename(embryo_path)
        data = new_data()
        data['_via_settings']['core']['default_filepath'] = f'{embryo_path}/'
        data['_via_settings']['project']['name'] = embryo
        plane_paths = glob.glob(os.path.join(embryo_path, '*'))
        for plane_path in plane_paths:
            filename = os.path.basename(plane_path)
            data['_via_img_metadata'][filename] = {
                'filename': filename,
                #'size': 696969, #TODO: extract size
                'regions': [],
                'file_attributes': {}
            }
        json.dump(data, open(f'{embryo}.json', 'w+'))
