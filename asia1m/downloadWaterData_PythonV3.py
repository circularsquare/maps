import urllib.request, sys, getopt, os
def main(argv):
   DESTINATION_FOLDER = argv[0]
   if (DESTINATION_FOLDER[-1:]!="/"):
      DESTINATION_FOLDER = DESTINATION_FOLDER + "/"
   if not os.path.exists(DESTINATION_FOLDER):
      print("Creating folder " + DESTINATION_FOLDER)
      os.makedirs(DESTINATION_FOLDER)
   DATASET_NAME = argv[1]
   longs = [str(e) + "E" for e in range(30,130,10)]
   lats = [str(s) + "S" for s in range(10,0,-10)]
   lats.extend([str(n) + "N" for n in range(0,70,10)])
   fileCount = len(longs)*len(lats)
   counter = 1
   for lng in longs:
      for lat in lats:
        filename = DATASET_NAME+ "_" + str(lng) + "_" + str(lat) + "v1_4_2021.tif"
        if os.path.exists(DESTINATION_FOLDER + filename):
           print(DESTINATION_FOLDER + filename + " already exists - skipping")
        else:
           url = "http://storage.googleapis.com/global-surface-water/downloads2021/" + DATASET_NAME + "/" + filename
           code = urllib.request.urlopen(url).getcode()
           if (code != 404):
              print("Downloading " + url + " (" + str(counter) + "/" + str(fileCount) + ")")
              urllib.request.urlretrieve(url, DESTINATION_FOLDER + filename)
           else:
              print(url + " not found")
        counter += 1
if __name__ == "__main__":
   main(sys.argv[1:])

   