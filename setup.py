 #!/usr/bin/env python
 # -*- coding: utf-8 -*-

import os
from distutils.core import setup
import glob

setup(name='shape_learning',
      version='0.1',
      license='ISC',
      description='A library for learning shape models from user demonstration',
      author='Deanna Hood',
      author_email='deanna.m.hood@gmail.com',
      package_dir = {'': 'src'},
      packages=['shape_learning'],
      data_files=[('share/shape_learning/letter_model_datasets/uji_pen_chars2', glob.glob("share/letter_model_datasets/uji_pen_chars2/*")),
                  ('share/doc/shape_learning', ['AUTHORS', 'LICENSE', 'README.md'])]
      )
