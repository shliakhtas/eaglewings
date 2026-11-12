"""Фото й відео для демо детекторів (scripts/02_vision.py): крок vision сам скачує їх із Commons у data/raw/media/, детекції — на сторінки."""
# фото з Wikimedia Commons (CC BY-SA). Координати: з Commons, якщо є; інакше — приблизно за описом.
PHOTOS = [
    dict(file="track_dadia.jpg", title="Лісова дорога в Дадья", lat=41.1190, lon=26.1985, approx=True,
         author="Pjposullivan", license="CC BY-SA 3.0", url="https://commons.wikimedia.org/wiki/File:Track_in_Dadia_Forest.JPG"),
    dict(file="visitor_centre.jpg", title="Інформаційний центр парку", lat=41.1279, lon=26.2232, approx=True,
         author="DocWoKav", license="CC BY-SA 4.0", url="https://commons.wikimedia.org/wiki/File:Dadia,_visitor_information_centre.jpg"),
    dict(file="visitor_centre_2.jpg", title="Інформаційний центр (панорама)", lat=41.1281, lon=26.2236, approx=True,
         author="DocWoKav", license="CC BY-SA 4.0", url="https://commons.wikimedia.org/wiki/File:Dadia,_visitor_information_centre_2.jpg"),
    dict(file="hotel_dadia.jpg", title="Готель біля парку", lat=41.1284, lon=26.2226, approx=True,
         author="DocWoKav", license="CC BY-SA 4.0", url="https://commons.wikimedia.org/wiki/File:Hotel_Dadia_National_Park.jpg"),
    dict(file="visitor_centre_3.jpg", title="Візит-центр (2008)", lat=41.1277, lon=26.2229, approx=True,
         author="Pjposullivan", license="CC BY-SA 3.0", url="https://commons.wikimedia.org/wiki/File:Dadia_forest_visitor_centre.JPG"),
    dict(file="forest_view.jpg", title="Вид на ліс з дороги", lat=41.1330, lon=26.2050, approx=True,
         author="Pjposullivan", license="CC BY-SA 3.0", url="https://commons.wikimedia.org/wiki/File:Dadia_forest_view.JPG"),
    dict(file="vulture_viewpoint.jpg", title="Оглядовий пункт годівлі грифів", lat=41.0413, lon=26.2583, approx=False,
         author="Héctor Ochoa 'Robot8A'", license="CC BY-SA 4.0", url="https://commons.wikimedia.org/wiki/File:Dadia,_July_2017,_photo_9.jpg"),
    dict(file="panoramio.jpg", title="Ліс Дадья (Panoramio)", lat=41.1125, lon=26.2408, approx=False,
         author="Panoramio user", license="CC BY-SA 3.0", url="https://commons.wikimedia.org/wiki/File:Greece_-_Dadia_forest_-_panoramio.jpg"),
]
VIDEO = dict(file="fire_karst_2022.webm", title="Камера-вежа №2 (демо: запис пожежі в Карсті, 2022)",
             cam_lat=41.0905, cam_lon=26.2010, view_km=2.5, bearing=200,
             author="Bojan Marušič", license="CC BY-SA 4.0",
             url="https://commons.wikimedia.org/wiki/File:Po%C5%BEar_na_italijanski_strani.webm")
