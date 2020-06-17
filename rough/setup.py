import setuptools
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


class BuildExtension(setuptools.Command):
    description = DistUtilsBuildExt.description
    user_options = DistUtilsBuildExt.user_options
    boolean_options = DistUtilsBuildExt.boolean_options
    help_options = DistUtilsBuildExt.help_options

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
        'extension.compute_overlap',
        ['extension/compute_overlap.pyx']
    ),
]


setuptools.setup(
    name='Udacity Capstone Project',
    version='0.1.0',
    description='Udactity Capstone Project',
    url='https://github.com/arturo-ai/anchor-conf-generator',
    author='Sardhendu Mishra',
    author_email='sardhendu@arturo.ai',
    maintainer='Sardhendu Mishra',
    maintainer_email='sardhendu@arturo.ai',
    cmdclass={'build_ext': BuildExtension},
    packages=setuptools.find_packages(),
    install_requires=[
        'Pillow==2.2.1',
        'keras==2.0.8',
        'tensorflow==1.3.0',
        'tqdm==4.46.1'
    ],
    ext_modules=extensions,
    setup_requires=["cython>=0.28", "numpy>=1.14.0"]
)
