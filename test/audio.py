import unittest
from footprint.models import Audio
from footprint.models import Project

class TestAudio(unittest.TestCase):

    def test_init(self):
        cache_folder = '/cache/test'
        filename = '/dataset/YTCdataset/letitbe/test.mp3'
        bucket_name = 'test'
        audio = create_audio(filename, cache_folder, bucket_name)
        self.assertEqual(audio.project.cache_folder, cache_folder)
        self.assertEqual(audio.filename, filename)
        self.assertEqual(audio.bucket, bucket_name)

    def test_load_signal_from_file(self):
        filename = '/dataset/YTCdataset/letitbe/test.mp3'
        audio = create_audio(filename)
        audio._full_load()
        self.assertEqual(len(audio.y) > 10, True)
        self.assertEqual(audio.sr > 10, True)
        self.assertEqual(audio.tempo > 1, True)
        self.assertEqual(len(audio.beats) > 1, True)

    def test_cache_filename(self):
        filename = '/dataset/YTCdataset/letitbe/test.mp3'
        cache_folder = '/cache/test'
        audio = create_audio(filename, cache_folder)
        exp = '/cache/test/dataset/YTCdataset/letitbe/test.mp3/audio.hdf5'
        f = audio.cache_filename()
        self.assertEqual(f, exp)

    def test_cache_dump(self):
        filename = '/dataset/YTCdataset/letitbe/test.mp3'
        cache_folder = '/cache/test'
        audio = create_audio(filename, cache_folder)
        audio.clean_cache()
        load_audio_from_file(audio)
        ori_len_y = len(audio.y)
        ori_sr_orig = audio.sr
        audio.cache_dump()
        audio = None
        audio2 = create_audio(filename, cache_folder)
        audio2._cache_load()
        self.assertEqual(len(audio2.y), ori_len_y)
        self.assertEqual(audio2.sr, ori_sr_orig)

def create_audio(filename, cache_folder=None, bucket_name=None):
    project = Project(cache_folder=cache_folder)
    return Audio(filename, project, bucket=bucket_name)

def load_audio_from_file(audio):
    audio._full_load()


if __name__ == '__main__':
    unittest.main()

#      import code; code.interact(local=dict(globals(), **locals()))
