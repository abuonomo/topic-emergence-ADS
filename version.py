import os
from setuptools_scm import get_version

version = get_version(root=os.path.dirname(os.path.abspath(__file__)))
version = version.split('.dev')[0]
print(version)