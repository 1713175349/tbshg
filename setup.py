import distutils.command.build as _build
import os
import sys
from distutils import spawn
from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages, Extension
from setuptools import setup

import re
import sys
import shutil
import platform

from subprocess import check_call, check_output, CalledProcessError
from distutils.version import LooseVersion

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import manifest_maker

# def extend_build():
#     class build(_build.build):
#         def run(self):
#             cwd = os.getcwd()
#             if spawn.find_executable('cmake') is None:
#                 sys.stderr.write("CMake is required to build this package.\n")
#                 sys.exit(-1)
#             _source_dir = os.path.split(__file__)[0]
#             _source_dir = os.path.join(_source_dir, 'tbshg')
#             print(os.listdir(_source_dir))
#             _build_dir = os.path.join(_source_dir, 'build_setup_py')
#             _prefix = get_python_lib()
#             try:
#                 cmake_configure_command = [
#                     'cmake',
#                     '-H{0}'.format(_source_dir),
#                     '-B{0}'.format(_build_dir),
#                     '-DCMAKE_INSTALL_PREFIX={0}'.format(_prefix),
#                     '-DCMAKE_BUILD_TYPE:STRING=Release',
#                 ]
#                 _generator = os.getenv('CMAKE_GENERATOR')
#                 if _generator is not None:
#                     cmake_configure_command.append('-G{0}'.format(_generator))
#                 spawn.spawn(cmake_configure_command)
#                 spawn.spawn(
#                     ['cmake', '--build', _build_dir, '--target', 'install'])
#                 os.chdir(cwd)
#             except spawn.DistutilsExecError:
#                 sys.stderr.write("Error while building with CMake\n")
#                 sys.exit(-1)
#             _build.build.run(self)

#     return build


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake not found. Version 3.1 or newer is required")

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < "3.1.0":
            raise RuntimeError("CMake 3.1 or newer is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print("Building extension: " + ext.name)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                      "-DPYTHON_EXECUTABLE=" + sys.executable]
        # cmake_args += ["-DPB_WERROR=" + os.environ.get("PB_WERROR", "OFF"),
        #                "-DPB_TESTS=" + os.environ.get("PB_TESTS", "OFF"),
        #                "-DPB_NATIVE_SIMD=" + os.environ.get("PB_NATIVE_SIMD", "ON"),
        #                "-DPB_MKL=" + os.environ.get("PB_MKL", "OFF"),
        #                "-DPB_CUDA=" + os.environ.get("PB_CUDA", "OFF")]

        cfg = os.environ.get("PB_BUILD_TYPE", "Release")
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            cmake_args += ["-A", "x64" if sys.maxsize > 2**32 else "Win32"]
            build_args += ["--", "/v:m", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            if "-j" not in os.environ.get("MAKEFLAGS", ""):
                parallel_jobs = 2 if not os.environ.get("READTHEDOCS") else 1
                build_args += ["--", "-j{}".format(parallel_jobs)]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DCPB_VERSION=\\"{}\\"'.format(env.get("CXXFLAGS", ""),
                                                             self.distribution.get_version())

        def build():
            os.makedirs(self.build_temp, exist_ok=True)
            print("build_temp:::::::",self.build_temp)
            print("sourcedir:::::::",ext.sourcedir)
            print("cmake_args:::::::",cmake_args)
            print("build_args:::::::",build_args)
            check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        try:
            build()
        except CalledProcessError:  # possible CMake error if the build cache has been copied
            shutil.rmtree(self.build_temp)  # delete build cache and try again
            build()

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.rst')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

_this_package = 'tbshg'

version = {}
with open(os.path.join(_here, 'version.py')) as f:
    exec(f.read(), version)

setup(
    name=_this_package,
    install_requires=["numpy","mpi4py"],
    version=version['__version__'],
    description='tbshg',
    long_description=long_description,
    author='zln',
    author_email='zln',
    url='none',
    license='MIT',
    packages=["tbshg", "tbshg.utils"],
    # py_modules=["a.py"],
    ext_modules=[CMakeExtension("tbshg.tbshg_core")],
    #py_modules=[_this_package],
    python_requires=">=3.5",
    include_package_data=True,
    scripts=["bin/calculateshg","bin/calculatelinear","bin/plotshgresult","bin/generatewcentxyz","bin/plotlinearresult"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    # cmdclass={'build': extend_build()}
    cmdclass={'build_ext': CMakeBuild}
    )