from distutils.core import setup
setup(
  name = 'mlinference',
  packages = ['MLinference', 'MLinference.strategies.auxfunc','MLinference.strategies', 'MLinference.architectures', 'MLinference.architectures.maskrcnn', 'MLinference.architectures.kerasClassifiers'],
  version = '0.0.1',
  license= '',
  description = 'Packages for inference of Machine Learning models with explicit model protocol',
  author = 'Juan Carlos Arbelaez',
  author_email = 'juanarbelaez@vaico.com.co',
  url = 'https://jarbest@bitbucket.org/jarbest/mlinference.git',
  download_url = 'https://bitbucket.org/jarbest/mlinference/get/master.tar.gz',
  keywords = ['vaico', 'common', 'ml', 'computer vision', 'machine learning'],
  install_requires=['numpy', 'opencv-python', 'MLgeometry', 'MLcommon', 'opencv-contrib-python', 'tensorflow==2.2', 'Keras==2.3.1'],
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
