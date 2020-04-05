from setuptools.extension import Extension
from setuptools import setup, find_packages, Command
from os import path
from io import open
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requires = [r for r in f.readlines() if not r.startwith('#')]

class BuildExtension(Command):
    description     = DistUtilsBuildExt.description
    user_options    = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options    = DistUtilsBuildExt.help_options

    def __init__(self, *args, **kwargs):
        from setuptools.command.build_ext import build_ext as SetupToolsBuildExt

        # Bypass __setatrr__ to avoid infinite recursion.
        self.__dict__['_command'] = SetupToolsBuildExt(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._command, name)

    def __setattr__(self, name, value):
        setattr(self._command, name, value)

    def initialize_options(self, *args, **kwargs):
        return self._command.initialize_options(*args, **kwargs)

    def finalize_options(self, *args, **kwargs):
        ret = self._command.finalize_options(*args, **kwargs)
        import numpy
        self.include_dirs.append(numpy.get_include())
        return ret

    def run(self, *args, **kwargs):
        return self._command.run(*args, **kwargs)


extensions = [
    Extension(
        'efficientdet.utils.compute_overlap',
        ['efficientdet/utils/compute_overlap.pyx']
    ),
]

setup(
    name='efficientdet',

    version='0.1',
    description='EfficientDet Tensorflow 2.0 implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/Guillem96/efficientdet-tf',

    author='Guillem96 - Guillem Orellana Trullols',
    author_email='guillem.orellana@gmail.com', 

    cmdclass={'build_ext': BuildExtension},

    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data Scientists - Deep Learning Engineers',
        'Topic :: Deep Learning',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
    ],

    keywords='object-detection efficientdet bifpn',

    packages=find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.6',

    install_requires=requires,
    ext_modules=extensions,
    # TODO: Entrypoint to train and evaluate
    project_urls={ 
        'Source': 'https://github.com/Guillem96/efficientdet-tf',
    },
)