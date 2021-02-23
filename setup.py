# -*- coding: utf-8 -*-

import codecs
import os.path

from setuptools import find_packages, setup


################################################################################

def read_rel(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read_rel(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("unable to find version string.")


################################################################################


AUTHORS = "Guillem Orellana Trullols (https://github.com/Guillem96)\n"
AUTHORS += "Josep Pon Farreny (https://github.com/jponf)"

EMAILS = ""

# Short description
DESCRIPTION = ("This repository contains the implementation of the de-noising "
               "model described in \"DeepErase: Weakly Supervised Ink Artifact "
               "Removal in Document Text Images\" "
               "(https://arxiv.org/abs/1910.07070).")

# Long description
with open("README.md", "rt") as f:
    README = f.read()

# Requirements
with open("requirements.txt", "rt") as f:
    REQUIREMENTS = [x for x in map(str.strip, f.read().splitlines())
                    if x and not x.startswith("#")]

KEYWORDS = ["GAN", "Denoise", "text", "Tensorflow", "deeperase", "UNet"]


###############################################################################

setup(
    name='deeperase',
    version=get_version(os.path.join("deeperase", "__init__.py")),
    description=DESCRIPTION,
    long_description=README,
    author=AUTHORS,
    author_email=EMAILS,
    url="https://github.com/Guillem96/deeperase",
    license="",
    keywords=KEYWORDS,
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    package_data={},
    platforms='any',
    zip_safe=True,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
)