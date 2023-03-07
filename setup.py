from distutils.util import convert_path

from setuptools import find_packages, setup

VERSION = "0.0.1.dev0"

# Install requires
base_packages_path = convert_path("requirements.txt")
with open(base_packages_path) as base_packages_path_file:
    base_packages = base_packages_path_file.read().splitlines()


# Install test-requires
test_packages_path = convert_path("requirements_dev.txt")
with open(test_packages_path) as test_packages_path_file:
    test_packages = test_packages_path_file.read().splitlines()


# METADATA
DISTNAME = "riks_ds_utils"
DESCRIPTION = "short description of project."

with open("./README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = "Riksarkivet"
LICENSE = "MIT"
PROJECT_URL = "https://github.com/Borg93/riks_ds_utils"

def setup_package():
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        license=LICENSE,
        url=PROJECT_URL,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 1 - Planning",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Operating System :: Unix",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows"
        ],
        python_requires=">=3.9",
        install_requires=base_packages,
        include_package_data=True,
        package_dir={"": "src"},
        packages=find_packages(where="src", include=["riks_ds_utils*"]),
        package_data={DISTNAME: ["py.typed"]},
        extras_require={"tests": test_packages},
        data_files=[("requirements", ["requirements.txt", "requirements_dev.txt"])],
    )

if __name__ == "__main__":
    setup_package()
