#!/usr/bin/env bash
# This script gets run on TravisCI to deploy new releases. It gets triggered
# on tagged versions only, so make sure to tag when releasing!
set -ex

python maint/build.py
anaconda -t $ANACONDA_TOKEN upload build/**/$PKG_NAME-*.tar.bz2
