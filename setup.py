from setuptools import setup
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
  long_description_content_type="text/markdown",
  name = 'AutoWave',         
  packages = ['AutoWave'],
  version = '0.3',
  license='MIT',        
  description = 'AutoWave is a complete audio automatic classification package with plottings, audio analysis, data loading, and other capabilities.',
  long_description=README,
  author = 'Nilesh Verma',                   
  author_email = 'me@nileshverma.com',     
  url = 'https://github.com/Autowave/Autowave',
  download_url = 'https://github.com/Autowave/Autowave/archive/refs/tags/v0.3.tar.gz',    
  keywords = ['AutoWave', 'audio automatic classification',],   
  install_requires=[ 
      'numpy==1.19.5',
      'SoundFile==0.10.3.post1',
      'xgboost==1.1.1',
      'pandas==1.0.5',
      'pydub==0.25.1',
      'librosa==0.8.0',
      'tqdm==4.56.0',
      'matplotlib==3.2.2',
      'scipy==1.4.1',
      'lightgbm==2.3.1',
      'scikit_learn==0.24.2',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
