"""Python extension of the MLPol plugin. Built by plugin/build.sh."""
import os
from setuptools import Extension, setup

openmm_dir = os.environ['OPENMM_DIR']
plugin_dir = os.environ['MLPOL_INSTALL']
ext = Extension('_mlpolplugin', sources=['MLPolPluginWrapper.cpp'],
                libraries=['OpenMM', 'OpenMMAmoeba', 'MLPolPlugin'],
                include_dirs=[os.path.join(openmm_dir, 'include'), os.path.join(plugin_dir, 'include')],
                library_dirs=[os.path.join(openmm_dir, 'lib'), os.path.join(plugin_dir, 'lib')],
                runtime_library_dirs=[os.path.join(plugin_dir, 'lib'), os.path.join(openmm_dir, 'lib')],
                extra_compile_args=['-std=c++17'])
setup(name='mlpolplugin', version='0.1', py_modules=['mlpolplugin'], ext_modules=[ext])
