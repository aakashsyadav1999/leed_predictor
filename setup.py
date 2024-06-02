from setuptools import find_packages,setup
from typing import List

HYPEN_E_NOT = '-e .'

def get_requirements(file_path:str) ->List[str]:
    
    '''
    this function will return the list of requirements
    '''

    requirements = []

    with open(file_path) as file_obj:

        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"") for req in requirements]

        if HYPEN_E_NOT in requirements:
            requirements.remove(HYPEN_E_NOT)
        
    return requirements

setup(
    name = "leed_predictor",
    version = '0.0.1',
    author = 'Aakash Yadav',
    author_email = 'aakashsyadav1999@gmail.com',
    packages = find_packages(),
    include_package_data=True,
    install_requires = get_requirements('requirements.txt')
)