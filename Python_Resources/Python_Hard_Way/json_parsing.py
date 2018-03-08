import json
import pprint

with open('centro_platform_bi_data.json') as data_file:
    data = json.load(data_file)['agencies']
data[0]['agency']
# pprint.pprint(data)
#     columns = list(data)
# print(columns[1])
# class DictQuery(dict):
#     def get(self, path, default = None):
#         keys = path.split("/")
#         val = None
#
#         for key in keys:
#             if val:
#                 if isinstance(val, list):
#                     val = [ v.get(key, default) if v else None for v in val]
#                 else:
#                     val = val.get(key, default)
#             else:
#                 val = dict.get(self, key, default)
#
#             if not val:
#                 break;
#
#         return val
# for item in data:
#     print DictQuery(data).get("agencies/id")
# # pprint(data)
# # pprint(data['agencies'][1]['brands'])
#
# def get_keys(dl, keys_list):
#     if isinstance(dl, dict):
#         keys_list += dl.keys()
#         map(lambda x: get_keys(x, keys_list), dl.values())
#     elif isinstance(dl, list):
#         map(lambda x: get_keys(x, keys_list), dl)
# keys_list = []
# d = data
# get_keys(d, keys_list)
# print list(set(keys_list))
#
# print data[0]['rate_type']
# # agencies = data['agencies']
# # print "Value : %s" % agencies.keys()
