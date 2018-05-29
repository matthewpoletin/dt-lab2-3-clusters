#!/usr/bin/env python3
# -*- coding: utf-8 -*

import sys
import os
import csv
import json
import logging
import urllib.parse, urllib.request

from settings import config
from cluster import clusterize

include_mpn = True
include_manufacturer = True
limit = 100


def load(amount):
    """
    Loads information on given subject
    :param amount: Number of items to load
    :return: [{}, {}, ...]
    """
    responses = []
    for counter in range(0, amount, limit):
        logging.info("Making request # {}".format(counter))
        url = "http://octopart.com/api/v3/parts/search"
        url += "?apikey=" + config['octopart']['api_key']
        args = [
            ('q', 'mcu'),
            ('include[]', 'specs'),
            ('start', counter),
            ('limit', limit),
            ('hide[]', 'offers'),
            ('hide[]', 'facet'),
            ('hide[]', 'filter'),
            ('hide[]', 'msec'),
            ('hide[]', 'stats'),
        ]
        url += '&' + urllib.parse.urlencode(args)
        data = urllib.request.urlopen(url).read()
        search_response = json.loads(data)
        responses.append(search_response)
    return responses


def save(filename, data):
    """
    Saves list of dicts to csc file
    :param filename: Name of the file
    :param data: [{}, {}, ...]
    :return:
    """
    keys = []
    for item in data:
        data_keys = item.keys()
        for data_key in data_keys:
            if data_key not in keys:
                keys.append(data_key)
    with open(os.path.join('data', filename), 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, dialect=csv.excel)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def api():
    responses = load(1000)
    print("Founded {}".format(responses[0]['hits']))
    result = []
    for response in responses:
        for item in response['results']:
            item_specs = {}
            if include_mpn is True:
                item_specs['mpn'] = item['item']['mpn']
            if include_manufacturer is True:
                item_specs['manufacturer'] = item['item']['manufacturer']['name']
            for spec in item['item']['specs']:
                if type(item['item']['specs'][spec]['value']) is list and len(item['item']['specs'][spec]['value']) > 0:
                    item_specs[spec] = item['item']['specs'][spec]['value'][0]
            # else:
            # print(item['item']['specs'][spec]['value'])
            result.append(item_specs)
    # print(result)
    save('result.csv', result)


def main(argv):
    """
    Главная функция
    :param argv: Command line arguments
    :return:
    """
    # Загружаем данные по api
    api()
    # Определяем ключи, с которыми работать
    keys = [
        'clock_speed',
        'flash_memory_bytes',
        'memory_size',
        'memory_size',
        'number_of_bits',
        'pin_count',
        'ram_bytes',
        'supply_voltage_dc',
    ]
    clusterize('result.csv', keys)


if __name__ == '__main__':
    main(sys.argv)
