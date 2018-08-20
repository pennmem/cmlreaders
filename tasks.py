import glob
import os
import platform
import shutil
import sys

from invoke import task


@task
def clean(c):
    """Clean the build directory."""
    print("Removing build dir")
    try:
        shutil.rmtree("build")
        os.mkdir("build")
    except OSError:
        pass


@task(pre=[clean])
def build(c, pyver=None, convert=True):
    """Build a conda package.

    :param pyver: python version to build for (current interpreter version by
        default)
    :param convert: convert to other platforms after building (default: True)

    """
    if pyver is None:
        pyver = ".".join([str(v) for v in sys.version_info[:2]])

    cmd = [
        "conda", "build",
        "--output-folder=build/",
        "--python", pyver,
    ]

    for chan in ["conda-forge", "pennmem"]:
        cmd += ["-c", chan]

    cmd += ["conda.recipe"]

    c.run(" ".join(cmd))

    if convert:
        os_name = {
            "darwin": "osx",
            "win32": "win",
            "linux": "linux"
        }[sys.platform]
        dirname = "{}-{}".format(os_name, platform.architecture()[0][:2])
        files = glob.glob("build/{}/*.tar.bz2".format(dirname))

        for filename in files:
            cmd = "conda convert {} -p all -o build/".format(filename)
            c.run(cmd)


@task(pre=[build])
def upload(c):
    """Upload packages to Anaconda Cloud."""
    for platform in ["linux-64", "osx-64", "win-32", "win-64"]:
        files = glob.glob("build/{}/*.tar.bz2".format(platform))
        cmds = ["anaconda upload -u pennmem {}".format(f) for f in files]
        for cmd in cmds:
            c.run(cmd)


@task
def test(c, rhino_root=None):
    """Run unit tests.

    :param rhino_root: path to rhino root directory; when not given, don't run
        tests requiring rhino

    """
    if rhino_root is None:
        c.run('pytest -m "not rhino" cmlreaders/')
    else:
        c.run("pytest --rhino-root={} cmlreaders/".format(rhino_root))
