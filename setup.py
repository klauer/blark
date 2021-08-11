import sys
from os import path

from setuptools import find_packages, setup

import versioneer

min_version = (3, 6)

if sys.version_info < min_version:
    error = """
blark does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *sys.version_info[:2], *min_version
    )
    sys.exit(error)


here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = open(path.join(here, "requirements.txt")).read().splitlines()

setup(
    name="blark",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="SLAC National Accelerator Laboratory",
    license="GPL",
    packages=find_packages(),
    description="Beckhoff TwinCAT IEC 61131-3 parsing tools",
    long_description=readme,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "blark = blark.__main__:main",
        ]
    },
    package_data={
        "blark": ["blark/iec.lark", "blark/iec.grammar"],
    },
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=requirements,
)
