# https://lopezyse.medium.com/computer-vision-object-detection-with-python-14b241f97fd8
#  .venv/bin/python Main.py

# https://huggingface.co/unsloth/Qwen2-VL-2B-Instruct-bnb-4bit

""" 
TO DO:
- Filterung Bilder (ohne Videos)
- Auswahl Ordner
- Runterkonvertieren große Bilder
"""


import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image


import os
import os.path
import photoscript
import osxphotos


# Load the model
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)



def qwen_detect(image_path):
    image = [Image.open(image_path)] #can also be used with PIL.Image.Image objects
    prompt = "Describe this image." # Lange Beschreibung
    prompt = "Give this image a long title in german!"

    formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
    )
    output = generate(model, processor, formatted_prompt, image, verbose=False)
    return output
    
db = os.path.expanduser("/Users/stephan/Pictures/Fotomediathek_ab_2025.photoslibrary")
photosdb = osxphotos.PhotosDB(db)

# Ausgabe infos
print(photosdb.keywords)
print(photosdb.persons)
print(photosdb.albums)

print(photosdb.keywords_as_dict)
print(photosdb.persons_as_dict)
print(photosdb.albums_as_dict)

# filter photo
photos = photosdb.photos()

for p in photos[:3]:
    print("------------------------------")
    print(
        p.uuid,"\n",
        "Dateiname:",p.filename, "\n",
        "Keywords",p.keywords,"\n",
        "Titel:", p.title,"\n",
        "Beschreibung:", p.description,"\n",
        "Pfad:",p.path,
        
    )
    if p.path is not None:
        try:
            Beschreibung = qwen_detect(p.path)
            print("Bildbeschreibung:", Beschreibung)
            p = photoscript.Photo(p.uuid)
            
            p.title = Beschreibung
            # p.description = Beschreibung # geht nicht -> Fehlermeldung
            #p.keywords = p.keywords + ["Beschreibung: "+Beschreibung]
        except Exception as e:
            print("Problem: ",e)
            


    


"""
uuid: 6CD4E1D9-07A7-4866-B2D9-5E022BCD020E
filename: DSC01863.dng
original_filename: DSC01863.dng
date: '2025-02-26T17:12:45.616000+01:00'
description: null
title: null
keywords: []
albums:
- "K\xF6ln Februrar"
persons: []
path: "/Users/stephan/Desktop/2025/Ko\u0308ln Februrar/DSC01863.dng"
ismissing: false
hasadjustments: false
external_edit: false
favorite: false
hidden: false
latitude: 50.9936
longitude: 7.057541333333333
path_edited: null
shared: false
isphoto: true
ismovie: false
uti: com.adobe.raw-image
burst: false
live_photo: false
path_live_photo: null
iscloudasset: false
incloud: null
date_modified: null
portrait: false
screenshot: false
screen_recording: false
slow_mo: false
time_lapse: false
hdr: false
selfie: false
panorama: false
has_raw: false
uti_raw: null
path_raw: null
place: !!python/object:osxphotos.placeinfo.PlaceInfo5
  _bplist: !!binary |
    YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8Q
    D05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGvECwLDCEpNT9AQUhNTlNUVVpbYGFiZ2hpbm9wdXZ3
    e36ChoeLn6ChoqOkpaipqlUkbnVsbNoNDg8QERITFBUWFxgZGhscHRofIFZpc0hvbWVWJGNsYXNz
    XXBvc3RhbEFkZHJlc3NfEBZjb21wb3VuZFNlY29uZGFyeU5hbWVzW2NvdW50cnlDb2RlXWFkZHJl
    c3NTdHJpbmdXdmVyc2lvbl1jb21wb3VuZE5hbWVzXxASZ2VvU2VydmljZVByb3ZpZGVyV21hcEl0
    ZW0IgCuAIYAAgB+AKRANgACAKoAC1A4iIyQlJicoXxAPZmluYWxQbGFjZUluZm9zXxAQc29ydGVk
    UGxhY2VJbmZvc18QEGJhY2t1cFBsYWNlSW5mb3OAIIAdgAOAHNIqDis0Wk5TLm9iamVjdHOoLC0u
    LzAxMjOABIAIgAqADYAPgBKAFYAYgBvVNg43ODk6Ozw9Pl8QEWRvbWluYW50T3JkZXJUeXBlVGFy
    ZWFUbmFtZVlwbGFjZVR5cGUQAIAHIwAAAAAAAAAAgAWABm8QXQBMAGEAbgBkAHMAYwBoAGEAZgB0
    AHMAcwBjAGgAdQB0AHoAZwBlAGIAaQBlAHQAIABEAGUAbABsAGIAcgD8AGMAawBlAHIAIABXAGEA
    bABkACwAIAB2AG8AcgBnAGUAbABhAGcAZQByAHQAZQAgAEYAcgBlAGkAcgDkAHUAbQBlACAAdQBu
    AGQAIAB2AGUAcgBiAGkAbgBkAGUAbgBkAGUAIABHAHIA/ABuAGIAZQByAGUAaQBjAGgAZRAI0kJD
    REVaJGNsYXNzbmFtZVgkY2xhc3Nlc18QIlBMUmV2R2VvTWFwSXRlbUFkZGl0aW9uYWxQbGFjZUlu
    Zm+iRkdfECJQTFJldkdlb01hcEl0ZW1BZGRpdGlvbmFsUGxhY2VJbmZvWE5TT2JqZWN01TYONzg5
    OjtKSz6AByNBT3jdAAAAAIAJgAZvEF0ATABhAG4AZABzAGMAaABhAGYAdABzAHMAYwBoAHUAdAB6
    AGcAZQBiAGkAZQB0ACAARABlAGwAbABiAHIA/ABjAGsAZQByACAAVwBhAGwAZAAsACAAdgBvAHIA
    ZwBlAGwAYQBnAGUAcgB0AGUAIABGAHIAZQBpAHIA5AB1AG0AZQAgAHUAbgBkACAAdgBlAHIAYgBp
    AG4AZABlAG4AZABlACAARwByAPwAbgBiAGUAcgBlAGkAYwBoAGXVNg43ODk6O1BRUoAHI0FknG0A
    AAAAgAuADGgARAD8AG4AbgB3AGEAbABkEAbVNg43ODk6O1dYUoAHI0GIz0MgAAAAgA6ADGcATQD8
    AGwAaABlAGkAbdU2Djc4OTo7XV5fgAcjQbgOLuAAAACAEIARZABLAPYAbABuEATVNg43ODk6O2Rl
    ZoAHI0H7VacAAAAAgBOAFGQASwD2AGwAbhAD1TYONzg5OjtrbG2AByNCH6avIAAAAIAWgBdfEBNO
    b3JkcmhlaW4tV2VzdGZhbGVuEALVNg43ODk6O3JzdIAHI0JWHRXAAAAAgBmAGltEZXV0c2NobGFu
    ZBAB0kJDeHleTlNNdXRhYmxlQXJyYXmjeHpHV05TQXJyYXnSKg58NKCAG9IqDn80oYCAHoAb1TYO
    Nzg5HTs8Gz6AB4AfgAZSREXSQkOIiV8QD1BMUmV2R2VvTWFwSXRlbaKKR18QD1BMUmV2R2VvTWFw
    SXRlbdqMjQ6Oj5CRkpOUlRqXmBqam5ydG1Zfc3RhdGVfEBFfZm9ybWF0dGVkQWRkcmVzc1VfY2l0
    eVdfc3RyZWV0W19wb3N0YWxDb2RlWF9jb3VudHJ5XxAWX3N1YkFkbWluaXN0cmF0aXZlQXJlYVxf
    c3ViTG9jYWxpdHlfEA9fSVNPQ291bnRyeUNvZGWAJYAAgCiAI4AAgCaAJ4AkgCKAH2cATQD8AGwA
    aABlAGkAbWQASwD2AGwAbmQASwD2AGwAbl8QE05vcmRyaGVpbi1XZXN0ZmFsZW5VNTEwNjlbRGV1
    dHNjaGxhbmTSQkOmp18QD0NOUG9zdGFsQWRkcmVzc6KmR28QdgBMAGEAbgBkAHMAYwBoAGEAZgB0
    AHMAcwBjAGgAdQB0AHoAZwBlAGIAaQBlAHQAIABEAGUAbABsAGIAcgD8AGMAawBlAHIAIABXAGEA
    bABkACwAIAB2AG8AcgBnAGUAbABhAGcAZQByAHQAZQAgAEYAcgBlAGkAcgDkAHUAbQBlACAAdQBu
    AGQAIAB2AGUAcgBiAGkAbgBkAGUAbgBkAGUAIABHAHIA/ABuAGIAZQByAGUAaQBjAGgAZQAsACAA
    NQAxADAANgA5ACAASwD2AGwAbgAsACAARABlAHUAdABzAGMAaABsAGEAbgBkVDc2MTjSQkOrrF8Q
    FFBMUmV2R2VvTG9jYXRpb25JbmZvoq1HXxAUUExSZXZHZW9Mb2NhdGlvbkluZm8ACAARABoAJAAp
    ADIANwBJAEwAUQBTAIIAiACdAKQAqwC5ANIA3gDsAPQBAgEXAR8BIAEiASQBJgEoASoBLAEuATAB
    MgE7AU0BYAFzAXUBdwF5AXsBgAGLAZQBlgGYAZoBnAGeAaABogGkAaYBsQHFAcoBzwHZAdsB3QHm
    AegB6gKnAqkCrgK5AsIC5wLqAw8DGAMjAyUDLgMwAzID7wP6A/wEBQQHBAkEGgQcBCcEKQQyBDQE
    NgRFBFAEUgRbBF0EXwRoBGoEdQR3BIAEggSEBI0EjwSaBJwEpQSnBKkEvwTBBMwEzgTXBNkE2wTn
    BOkE7gT9BQEFCQUOBQ8FEQUWBRgFGgUcBScFKQUrBS0FMAU1BUcFSgVcBXEFeAWMBZIFmgWmBa8F
    yAXVBecF6QXrBe0F7wXxBfMF9QX3BfkF+wYKBhMGHAYyBjgGRAZJBlsGXgdNB1IHVwduB3EAAAAA
    AAACAQAAAAAAAACuAAAAAAAAAAAAAAAAAAAHiA==
  _plrevgeoloc: !!python/object:osxphotos.placeinfo.PLRevGeoLocationInfo
    addressString: "Landschaftsschutzgebiet Dellbr\xFCcker Wald, vorgelagerte Freir\xE4\
      ume und verbindende Gr\xFCnbereiche, 51069 K\xF6ln, Deutschland"
    countryCode: DE
    mapItem: !!python/object:osxphotos.placeinfo.PLRevGeoMapItem
      sortedPlaceInfos:
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 0.0
        name: "Landschaftsschutzgebiet Dellbr\xFCcker Wald, vorgelagerte Freir\xE4\
          ume und verbindende Gr\xFCnbereiche"
        placeType: 8
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 4125114.0
        name: "Landschaftsschutzgebiet Dellbr\xFCcker Wald, vorgelagerte Freir\xE4\
          ume und verbindende Gr\xFCnbereiche"
        placeType: 8
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 10806120.0
        name: "D\xFCnnwald"
        placeType: 6
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 52029540.0
        name: "M\xFClheim"
        placeType: 6
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 403582688.0
        name: "K\xF6ln"
        placeType: 4
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 7337570304.0
        name: "K\xF6ln"
        placeType: 3
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 33985120256.0
        name: Nordrhein-Westfalen
        placeType: 2
        dominantOrderType: 0
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 379908980736.0
        name: Deutschland
        placeType: 1
        dominantOrderType: 0
      finalPlaceInfos:
      - !!python/object:osxphotos.placeinfo.PLRevGeoMapItemAdditionalPlaceInfo
        area: 0.0
        name: DE
        placeType: 8
        dominantOrderType: 13
    isHome: false
    compoundNames: null
    compoundSecondaryNames: null
    version: 13
    geoServiceProvider: '7618'
    postalAddress: !!python/object:osxphotos.placeinfo.CNPostalAddress
      _ISOCountryCode: DE
      _city: "K\xF6ln"
      _country: Deutschland
      _postalCode: '51069'
      _state: Nordrhein-Westfalen
      _street: null
      _subAdministrativeArea: "K\xF6ln"
      _subLocality: "M\xFClheim"
  _names: !!python/object/new:osxphotos.placeinfo.PlaceNames
  - []
  - - Deutschland
  - - Nordrhein-Westfalen
  - - "K\xF6ln"
  - - "K\xF6ln"
  - []
  - - "D\xFCnnwald"
    - "M\xFClheim"
  - []
  - - "Landschaftsschutzgebiet Dellbr\xFCcker Wald, vorgelagerte Freir\xE4ume und\
      \ verbindende Gr\xFCnbereiche"
    - "Landschaftsschutzgebiet Dellbr\xFCcker Wald, vorgelagerte Freir\xE4ume und\
      \ verbindende Gr\xFCnbereiche"
  - []
  - []
  - []
  - []
  - []
  - []
  - []
  - []
  - []
  - []
  _name: "Landschaftsschutzgebiet Dellbr\xFCcker Wald, vorgelagerte Freir\xE4ume und\
    \ verbindende Gr\xFCnbereiche, K\xF6ln, Nordrhein-Westfalen, Deutschland"
exif: ExifInfo(flash_fired=False, iso=500, metering_mode=5, sample_rate=None, track_format=None,
  white_balance=11, aperture=3.2, bit_rate=None, duration=None, exposure_bias=0.0,
  focal_length=35.0, fps=None, latitude=50.9936, longitude=7.057541333333333, shutter_speed=0.01,
  camera_make='SONY', camera_model='ILCE-6700', codec=None, lens_model='E 35mm F1.8
  OSS', date=datetime.datetime(2025, 2, 26, 17, 12, 45, 616000, tzinfo=zoneinfo.ZoneInfo(key='Europe/Berlin')),
  tzoffset=3600, tzname='Europe/Berlin')
score: ScoreInfo(overall=0.7353515625, curation=0.5, promotion=0.0, highlight_visibility=0.03816793893129771,
  behavioral=0.0, failure=-0.001953125, harmonious_color=0.10406494140625, immersiveness=0.00341796875,
  interaction=0.009999999776482582, interesting_subject=0.2110595703125, intrusive_object_presence=-0.033935546875,
  lively_color=-0.1600341796875, low_light=0.010986328125, noise=-0.02392578125, pleasant_camera_tilt=0.0003662109375,
  pleasant_composition=0.359375, pleasant_lighting=0.423583984375, pleasant_pattern=0.020751953125,
  pleasant_perspective=0.378173828125, pleasant_post_processing=0.02655029296875,
  pleasant_reflection=0.076904296875, pleasant_symmetry=0.003662109375, sharply_focused_subject=0.63916015625,
  tastefully_blurred=0.880859375, well_chosen_subject=-0.10015869140625, well_framed_subject=0.7685546875,
  well_timed_shot=0.059295654296875)
intrash: false
height: 6192
width: 4128
orientation: 8
original_height: 4128
original_width: 6192
original_orientation: 8
original_filesize: 27948216
"""