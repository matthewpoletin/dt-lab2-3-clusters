#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

configFile = open(os.path.join('config.json'))
config = (json.load(configFile))
