import os
import numpy as np
import h5py
import hashlib
from PIL import Image

import tensorflow as tf
from utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output')
# TFRecords.')
FLAGS = flags.FLAGS


def get_attrs(digit_struct_mat_file, index):
    """
    Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
    """
    attrs = {}
    f = digit_struct_mat_file
    item = f['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = f[item][key]
        values = [f[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs


def convert_to_tf_record(name):
	path_root_dir = os.path.join(FLAGS.data_dir, name)
	path_mat_file = os.path.join(path_root_dir, 'digitStruct.mat')
	path_output_file = os.path.join(FLAGS.output_dir, name + '.tfrecords')

	with tf.python_io.TFRecordWriter(path_output_file) as writer:
		with h5py.File(path_mat_file, 'r') as mat_file:
			path_image_files = tf.gfile.Glob(os.path.join(path_root_dir, '*.png'))
			total_files_count = len(path_image_files)

			truncated = []
			poses = []
			difficult = []
			for i, path_image_file in enumerate(path_image_files):

				filename = os.path.basename(path_image_file)
				name, _ = os.path.splitext(filename)
				index = int(name) - 1

				print("(%.2f%%) Processing image %s" % (i/total_files_count*100, filename))

				image = Image.open(path_image_file)
				encoded_image = np.array(image).tobytes()
				key = hashlib.sha256(encoded_image).hexdigest()
				
				attrs = get_attrs(mat_file, index)				

				xmin = np.array(attrs['left'])
				ymin = np.array(attrs['top'])
				xmax = xmin + np.array(attrs['width'])
				ymax = ymin + np.array(attrs['height'])

				classes = np.int0(attrs['label'])
				for i, c in enumerate(classes):
					if c == 10:
						classes[i] = 0

				classes_text = [str(c).encode('utf8') for c in classes]
				
				truncated.append(1)
				poses.append('Unspecified'.encode('utf8'))
				difficult.append(0)

				example = tf.train.Example(features=tf.train.Features(feature={
					'image/height': dataset_util.int64_feature(image.size[0]),
					'image/width': dataset_util.int64_feature(image.size[1]),
					'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
					'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
					'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
					'image/encoded': dataset_util.bytes_feature(np.array(image).tobytes()),
					'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
					'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
					'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
					'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
					'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
					'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
					'image/object/class/label': dataset_util.int64_list_feature(classes),
					'image/object/difficult': dataset_util.int64_list_feature(difficult),
					'image/object/truncated': dataset_util.int64_list_feature(truncated),
					'image/object/view': dataset_util.bytes_list_feature(poses)
				}))

				writer.write(example.SerializeToString())	


def main(_):
	for name in ['train', 'test', 'extra']:
		convert_to_tf_record(name)

if __name__ == '__main__':

	tf.app.run()