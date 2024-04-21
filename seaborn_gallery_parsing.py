import os

import requests
from bs4 import BeautifulSoup
from omegaconf import OmegaConf
from tqdm import tqdm

# %%


def match_func_image(tag):
    return (
        tag.name == "img"
        and tag.has_attr("src")
        and tag["src"].startswith("../_images/")
    )


def get_image(soup, out_folder):
    images = soup.find_all(match_func_image)

    image_urls = []

    for i, image in enumerate(images):
        image_url_relative = image["src"]
        image_url = "https://seaborn.pydata.org" + image_url_relative[2:]
        image_urls.append(image_url)

        img_response = requests.get(image_url, stream=True)

        if i == 0:
            image_name = "plot.png"
        else:
            image_name = "plot_{i}.png"

        image_path = os.path.join(out_folder, image_name)

        with open(image_path, "wb") as f:
            f.write(img_response.content)

    return image_urls


def get_code(soup, out_folder):
    code_block = soup.find("div", class_="highlight").get_text()

    code_path = os.path.join(out_folder, "plot_code.py")

    with open(code_path, "w") as f:
        f.write(code_block)

    return code_block


def get_gallary_links():
    base_url = "https://seaborn.pydata.org/examples/"
    gallery_url = base_url + "index.html"

    r = requests.get(gallery_url)
    soup = BeautifulSoup(r.text, "html.parser")

    plot_links = []

    for link in soup.find_all("a"):
        href = link.get("href")

        if href is not None and href.startswith("./") and href.endswith(".html"):
            plot_url = base_url + href[2:]
            plot_links.append(plot_url)

    return plot_links


# %%
if __name__ == "__main__":
    plot_links = get_gallary_links()

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)

    out_base_folder = config.seaborn_out_path

    for plot_url in tqdm(plot_links):
        plot_name = plot_url.split("/")[-1][:-5]

        out_folder = os.path.join(out_base_folder, plot_name)
        os.makedirs(out_folder, exist_ok=True)

        response = requests.get(plot_url)
        soup = BeautifulSoup(response.text, "html.parser")

        code_block = soup.find("div", class_="highlight").get_text()
        image_url = get_image(soup, out_folder)[0]
        get_code(soup, out_folder)
