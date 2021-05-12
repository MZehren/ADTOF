import logging
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve
import os
from bs4 import BeautifulSoup
import re
from pathlib import Path


def scrapIndex(rangeToParse=range(1, 100), path=""):
    """Download the custom charts from Rhythm Gaming world website

    Parameters
    ----------
    rangeToParse : range, optional
        The range of pages to download, by default range(1, 100)
    path : str, optional
        folder where to store the downloaded files, by default ""
    """
    for page in rangeToParse:
        logging.info("page :" + str(page))
        url = "https://db.c3universe.com/songs?page=" + str(page)
        soup = BeautifulSoup(urlopen(url), "html.parser")
        rows = soup.find("table", id="database").select("tbody tr")
        for track in range(0, len(rows), 2):
            link = rows[track].select("td:nth-of-type(2) a")[0]["href"]
            artist = rows[track].select("td:nth-of-type(3) a")[0].text[1:-1]
            title = rows[track].select("td:nth-of-type(4) a")[0].find("div", {"class", "todo-tasklist-item-title"}).text[1:-1]
            hasDrums = len(rows[track + 1].select("div:nth-of-type(3) span")) == 0
            fileName = artist + " - " + title
            if hasDrums:
                print(fileName, downloadCustom(link, path))


def pickLink(soup):
    """
    from all the versions that exists from a file, download the best one.
    """
    raise DeprecationWarning("pickLink is a deprecated method not useful with the current version of the dataset")
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


def downloadCustom(url, folderPath):
    """
    Download the annoation from the url and return True if succeeded

    Url: url of the page hosting the chart
    fodlerPath: path to the folder where the file has to be downloaded
    """

    try:
        Path(folderPath).mkdir(parents=True, exist_ok=True)
        if url is None:
            return False
        elif "c3universe.com/" in url:
            soup = BeautifulSoup(urlopen(url), "html.parser")
            downloadUrl = soup.find("a", {"class": "btn-warning"})["href"]
            fileName = downloadUrl.split("/")[-1]
            filePath = os.path.join(folderPath, fileName)
            if os.path.exists(filePath) == False:
                urlretrieve(quote(downloadUrl, ":/"), filePath)
            return True
        elif "mediafire.com/" in url:
            raise DeprecationWarning("Mediafire is not supported anymore")
            soup = BeautifulSoup(urlopen(url), "html.parser")
            downloadUrl = soup.find("a", {"aria-label": "Download file"})["href"]
            dl(downloadUrl)
            return True
    except Exception as e:
        logging.error("url not working: " + url)
        logging.error("exception raised: " + e)

    return False
