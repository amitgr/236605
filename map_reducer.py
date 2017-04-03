import csv
from common import Triplet
import re

BRAND_IDX = 8
GTIN_IDX = 12
MPN_IDX = 10

_regex = re.compile('^\\d+$')
def _is_mpn_valid(mpn):
  return not (' ' in mpn or 'n/a' in mpn.lower())
def _is_gtin_valid(gtin):
  return _regex.match(gtin)


def TripletsFromSqlDump():
  with open('E:\\Academic\\data_science\\sql_dump.txt', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    d = {}
    for line in reader:
      id = line[0]
      brand = line[BRAND_IDX].lower()
      mpn = line[MPN_IDX]
      gtin = line[GTIN_IDX]
      if any(map(lambda x: line[x+1] == '1' or line[x] == '?', [BRAND_IDX, MPN_IDX, GTIN_IDX])):
        continue
      if not _is_gtin_valid(gtin) or not _is_mpn_valid(mpn):
        continue
      d[id] = d.get(id, {})
      histogram = d[id]
      triplet = Triplet(GTIN=gtin, Brand=brand, MPN=mpn, result=None)
      histogram[triplet] = histogram.get(triplet, 0) + 1

    for key, values in d.items():
      import heapq
      largest_2 = heapq.nlargest(2, values, key=lambda t: values[t])
      largest = largest_2[0]
      if values[largest] == 1:
        continue
      if len(largest_2) > 1 and values[largest] == values[largest_2[1]]:
        continue
      for triplet in values.keys():
        yield Triplet(MPN=triplet.MPN, GTIN=triplet.GTIN, Brand=triplet.Brand, result=triplet == largest)


def main():
  positives = 0
  negatives = 0
  for triplet in TripletsFromSqlDump():
    if triplet.result:
      positives += 1
    else:
      negatives += 1
    # if (positives + negatives) % 100 == 0:
    #if (triplet.GTIN == '0042526263767'):
    if triplet.GTIN.isspace():
      print(triplet)

  print('positives ', positives)
  print('negatives ', negatives)

if __name__ == '__main__':
  main()

