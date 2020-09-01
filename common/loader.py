"""
Load shared models from zip architectures

JCA
Vaico
"""
import zipimport
import pickle
import logging
import zipfile
from sys import path as sys_path
from os.path import join

log = logging.getLogger(__name__)


def load_zip_model(model_path, zip_path):
    """Load .ml models where the architecture is in .zip file
    Install the requirements of the specific architecture to be loaded
    :param model_path: (str) file path to .ml model
    :param zip_path: (str) path to compressed architecture package
    :return: (MLcommon.AbcModel)
    """
    log.info('Loading model from: {}'.format(model_path))
    with open(model_path, 'rb') as handle:
        model_data = pickle.load(handle)
    target_name = model_data['conf']['architecture']

    log.info('Loading architecture module: {}'.format(zip_path))
    sys_path.append(join(zip_path, 'python'))

    importer = zipimport.zipimporter(zip_path)
    # Get architecture folder and module names
    arch_name = None
    module_name = None
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if 'python/{0}/{0}'.format(target_name).lower() in name.lower():
                _, arch_name, module_name = name.split('/')
                module_name = module_name.split('.')[0]
    log.info('Loading: {}/{}'.format(arch_name, module_name))

    model_module = importer.load_module(join('python', arch_name, module_name))

    # Search architecture class name in module
    for attr in dir(model_module):
        if attr.lower() == arch_name.lower():
            arch_class_name = attr
            break

    architecture = getattr(model_module, arch_class_name)
    return architecture.load(model_path)
