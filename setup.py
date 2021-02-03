

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepvog3D",
    version="1.1.1",
    author="Yuk-Hoi Yiu et al.",
    author_email="h.yiu@campus.lmu.de",
    description="Deep VOG for gaze estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pydsgz/DeepVOG",
    license="GNU General Public License v3.0",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    package_data = {
        'deepvog3D':['model3D/*.py', 'model3D/*.h5'],
        #'model_weights':['model/*.h5'],
        
    },
    python_requires='>=3.5',
    install_requires=['numpy>=1.12',
                      'scikit-video>=1.1.0',
                      'scikit-image>=0.14.0',
                      'monai=0.3.0',
                      'urwid',
                      'opencv-python'],

    
)
