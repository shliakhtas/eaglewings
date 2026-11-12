# Джерела

Дані й методи, на яких стоять числа дослідження «Ліси Фракії». Кожне посилання перевірено 12.11.2026: DOI — через Crossref або DataCite, сторінки без DOI — відкриттям сторінки. Те, що перевірити не вдалося, позначено «не перевірено».

## Дані

| Джерело | Що беремо | Де використано | Опис продукту |
|---|---|---|---|
| NASA FIRMS, VIIRS S-NPP 375 м | теплові точки 2021–2025, Греція | FD, FM, FG_day, FG_risk, FG_spread, AR, TP, CG, LG (виключення згарищ) | Schroeder W., Oliva P., Giglio L. et al. (2014). The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment. *Remote Sensing of Environment* 143, 85–96. [doi:10.1016/j.rse.2013.12.008](https://doi.org/10.1016/j.rse.2013.12.008) · [FIRMS](https://firms.modaps.eosdis.nasa.gov/) |
| NASA FIRMS, MODIS Collection 6 | теплові точки 2021–2025 | ті самі гілки | Giglio L., Schroeder W., Justice C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment* 178, 31–41. [doi:10.1016/j.rse.2016.02.054](https://doi.org/10.1016/j.rse.2016.02.054) |
| Hansen Global Forest Change v1.12 | втрата лісу за роками, лісистість 2000 | FD (еталон), LD, LM, LG, CP, FG_risk | Hansen M. C., Potapov P. V., Moore R. et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science* 342, 850–853. [doi:10.1126/science.1244693](https://doi.org/10.1126/science.1244693) |
| EFFIS Burnt Areas | контури згарищ | FM (еталон), LD/LG (виключення згарищ) | San-Miguel-Ayanz J., Schulte E., Schmuck G. et al. (2012). Comprehensive Monitoring of Wildfires in Europe: The European Forest Fire Information System (EFFIS). У *Approaches to Managing Disaster*, InTech. [doi:10.5772/28441](https://doi.org/10.5772/28441) · [EFFIS](https://forest-fire.emergency.copernicus.eu/) |
| Sentinel-2 L2A | літні композити, індекси NBR, NDMI, NDVI | FD, LD, LM | Drusch M., Del Bello U., Carlier S. et al. (2012). Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services. *Remote Sensing of Environment* 120, 25–36. [doi:10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026) |
| MODIS MOD13Q1 v061 | NDVI 250 м до й після пожежі | FD | Didan K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA LP DAAC. [doi:10.5067/MODIS/MOD13Q1.061](https://doi.org/10.5067/MODIS/MOD13Q1.061) |
| NASA OPERA DIST-ALERT-HLS v1 | порушення рослинності (через шар GIBS) | LM | Hansen M. (2024). OPERA Land Surface Disturbance Alert from Harmonized Landsat Sentinel-2 product (Version 1). NASA LP DAAC. [doi:10.5067/SNWG/OPERA_L3_DIST-ALERT-HLS_V1.001](https://doi.org/10.5067/SNWG/OPERA_L3_DIST-ALERT-HLS_V1.001) |
| NASA POWER | щоденна й погодинна погода в точці парку | FG_day, FG_spread | [документація POWER](https://power.larc.nasa.gov/docs/) |
| SRTM (через AWS Terrain Tiles) | висота й ухил | FG_risk, FG_spread, TP, CG | Farr T. G., Rosen P. A., Caro E. et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics* 45, RG2004. [doi:10.1029/2005RG000183](https://doi.org/10.1029/2005RG000183) |
| OpenStreetMap | межі парків, дороги, треки, села | усі гілки з місцем; CP (мережа доріг) | Haklay M., Weber P. (2008). OpenStreetMap: User-Generated Street Maps. *IEEE Pervasive Computing* 7(4), 12–18. [doi:10.1109/MPRV.2008.80](https://doi.org/10.1109/MPRV.2008.80) |

## Еталони детекторів

| Еталон | Де використано | Джерело |
|---|---|---|
| VEDAI | SD · машини згори | Razakarivony S., Jurie F. (2016). Vehicle detection in aerial imagery: A small target detection benchmark. *Journal of Visual Communication and Image Representation* 34, 187–203. [doi:10.1016/j.jvcir.2015.11.002](https://doi.org/10.1016/j.jvcir.2015.11.002) |
| D-Fire | CD · дим і вогонь | de Venâncio P. V. A. B., Lisboa A. C., Barbosa A. V. (2022). An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. *Neural Computing and Applications* 34, 15349–15368. [doi:10.1007/s00521-022-07467-z](https://doi.org/10.1007/s00521-022-07467-z) |
| ENA24 | CD · машини на фотопастках | Yousif H., Kays R., He Z. et al. (2019). Animal Scanner: Software for classifying humans, animals, and empty frames in camera trap images. *Ecology and Evolution* 9, 1578–1589. [doi:10.1002/ece3.4747](https://doi.org/10.1002/ece3.4747) |

## Методи й метрики

| Метод | Де використано | Першоджерело |
|---|---|---|
| DBSCAN | FM («за день»), XM, AR (ареали), застосунок (ареали) | Ester M., Kriegel H.-P., Sander J., Xu X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*, 226–231. [PDF](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) |
| ST-DBSCAN | FM (переможець), FD (оболонка груп точок), XM | Birant D., Kut A. (2007). ST-DBSCAN: An algorithm for clustering spatial–temporal data. *Data & Knowledge Engineering* 60, 208–221. [doi:10.1016/j.datak.2006.01.013](https://doi.org/10.1016/j.datak.2006.01.013) |
| BCubed F1 | FM, XM, AS (зведення в зони) | Amigó E., Gonzalo J., Artiles J., Verdejo F. (2009). A comparison of extrinsic clustering evaluation metrics based on formal constraints. *Information Retrieval* 12, 461–486. [doi:10.1007/s10791-008-9066-8](https://doi.org/10.1007/s10791-008-9066-8) |
| PAI, KDE-карти гарячих точок | AR | Chainey S., Tompson L., Uhlig S. (2008). The Utility of Hotspot Mapping for Predicting Spatial Patterns of Crime. *Security Journal* 21, 4–28. [doi:10.1057/palgrave.sj.8350066](https://doi.org/10.1057/palgrave.sj.8350066) |
| NBR, dNBR | FD (переможець), LD | Key C. H., Benson N. C. (2006). Landscape Assessment: Ground measure of severity, the Composite Burn Index; and Remote sensing of severity, the Normalized Burn Ratio. У *FIREMON*, USDA Forest Service RMRS-GTR-164-CD. [USGS](https://www.usgs.gov/publications/landscape-assessment-ground-measure-severity-composite-burn-index-and-remote-sensing) |
| Відносне падіння індексу (RdNBR, ідея для rdNDVI) | FD (відносне падіння NDVI) | Miller J. D., Thode A. E. (2007). Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR). *Remote Sensing of Environment* 109, 66–80. [doi:10.1016/j.rse.2006.12.006](https://doi.org/10.1016/j.rse.2006.12.006) |
| LightGBM | FG_day, FG_risk, FG_spread, LD, LG | Ke G., Meng Q., Finley T. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*. [proceedings](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) |
| Random Forest | FG_day, FG_risk | Breiman L. (2001). Random Forests. *Machine Learning* 45, 5–32. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) |
| PR-AUC як основна метрика при рідких подіях | FG_day | Saito T., Rehmsmeier M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE* 10, e0118432. [doi:10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432) |
| Блоковий бутстреп для CI | усі гілки (блоки днів, гексів, згарищ, прогонів) | Künsch H. R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. *The Annals of Statistics* 17, 1217–1241. [doi:10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265) |
| Задача максимального покриття (жадібне розміщення) | TP (вежі), CP (камери) | Church R., ReVelle C. (1974). The maximal covering location problem. *Papers of the Regional Science Association* 32, 101–118. [doi:10.1007/BF01942293](https://doi.org/10.1007/BF01942293) |
| YOLO11-OBB | SD (переможець), застосунок (машини на знімках) | Ultralytics YOLO11, програмне забезпечення. [документація](https://docs.ultralytics.com/models/yolo11/) |
| Grounding DINO | SD, CD, демо детекторів | Liu S., Zeng Z., Ren T. et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) |
| YOLO-World | демо детекторів на фотопастках (не міряно) | Cheng T., Song L., Ge Y. et al. (2024). YOLO-World: Real-Time Open-Vocabulary Object Detection. [arXiv:2401.17270](https://arxiv.org/abs/2401.17270) |
| H3 (гексагональна сітка) | FG_risk, FG_spread, AR, LG, TP, AS | [h3geo.org](https://h3geo.org/) |
| Індекс Нестерова, VPD, наближений FWI | FG_day (правила-базові лінії) | не перевірено: формули взято зі стандартних описів, першоджерело не відкривали |
| Логістична регресія | FG_day, FG_risk, FG_spread, LG | стандартний метод, першоджерело не наводимо |

## Інструменти

| Інструмент | Роль |
|---|---|
| [Prefect](https://www.prefect.io/) | порядок кроків потоку, версії прогону, звіти |
| [ClearML](https://clear.ml/) | запуски, датасети, моделі (лише з `--clearml`) |
| scikit-learn, LightGBM, GeoPandas, Shapely, rasterio | розрахунки |
