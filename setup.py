from distutils.core import setup
setup(
  name = 'mlinference',
  packages = ['MLinference', 'MLinference.common', 'MLinference.common.PredictionStrategies.auxfunc','MLinference.common.PredictionStrategies', 'MLinference.geometry', 'MLinference.geometry.geometries', 'MLinference.architectures'],
  version = '0.0.7',
  license= '',
  description = 'Packages for inference of Machine Learning models with explicit model protocol',
  author = 'Juan Carlos Arbelaez',
  author_email = 'juanarbelaez@vaico.com.co',
  url = 'https://jarbest@bitbucket.org/jarbest/mlinference.git',
  download_url = 'https://bitbucket.org/jarbest/mlinference/get/master.tar.gz',
  keywords = ['vaico', 'common', 'ml', 'computer vision', 'machine learning'],
  install_requires=['numpy', 'opencv-python'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ]
)
