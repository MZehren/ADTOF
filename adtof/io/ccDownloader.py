""" Module for downloading customcreators customs """
import logging
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve
import os
from bs4 import BeautifulSoup
import re


def scrapIndex(rangeToParse=range(20, 100), perPage=50, path="E:/ADTSets/adtof/raw/c3Download/"):
    for i in rangeToParse:
        logging.info("page :" + str(i))
        url = (
            "http://customscreators.com/index.php?/page/index.html?sort_col=rating_value&sort_order=desc&per_page="
            + str(perPage)
            + "&filters%5B18%5D%5B1%2C2%2C3%2C4%2C5%2C6%2C7%5D=1&filters%5B23%5D%5B1%2C2%2C3%2C4%2C5%2C6%2C7%5D=1&st="
            + str(i * perPage)
        )
        soup = BeautifulSoup(urlopen(url), "html.parser")
        for soupLinks in [row.findAll("td") for row in soup.findAll("tr", {"class": "dbrow"})]:
            name = soupLinks[1].find("a").text + "_" + soupLinks[2].find("a").text
            link = pickLink(soupLinks[5])
            print(link, downloadCustom(link, path, name))


def pickLink(soup):
    """
    from all the versions that exists from a file, download the best one.
    """
    types = ["RB 2x", "PS 2x", "RB", "PS"]
    titles = [
        "Download this custom for Xbox 360 (2x BASS PEDAL)",
        "Download this custom for Phase Shift (2x BASS PEDAL)",
        "Download this custom for Xbox 360",
        "Download this custom for Phase Shift",
    ]
    for i, title in enumerate(titles):
        link = soup.find("a", {"title": title})
        if link:
            return link["href"]  # , types[i]


def downloadCustom(url, folderPath, fileName):
    """
    Download the annoation from the url and return True if succeeded

    Url: url of the page hosting the chart
    fodlerPath: path to the folder where the file has to be downloaded
    fileName: name of the file to prevent downloading twice the same file
    /\*:?"<>|
    """
    fileName = re.sub('[\/\\\*\:\?"\<\>\|]+', "", fileName)  # fileName.replace("/", "").replace(":", "")

    def dl(downloadUrl):
        urlretrieve(quote(downloadUrl, ":/"), folderPath + fileName)

    try:
        if os.path.exists(folderPath + fileName) == True:
            return True
        elif url is None:
            return False
        elif "c3universe.com/" in url:
            soup = BeautifulSoup(urlopen(url), "html.parser")
            downloadUrl = soup.find("a", {"class": "btn-warning"})["href"]
            dl(downloadUrl)
            return True
        elif "mediafire.com/" in url:
            soup = BeautifulSoup(urlopen(url), "html.parser")
            downloadUrl = soup.find("a", {"aria-label": "Download file"})["href"]
            dl(downloadUrl)
            return True
    except:
        logging.error("url not working: " + url)

    return False


if __name__ == "__main__":
    scrapIndex()
