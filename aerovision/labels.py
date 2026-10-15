"""UC Merced land-use classes with human prompts, plus operational mapping used by downstream agents."""
CLASSES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest',
           'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
           'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

PROMPT_NAMES = {'agricultural': 'agricultural farmland', 'airplane': 'an airplane parked at an airport', 'baseballdiamond': 'a baseball diamond',
                'beach': 'a beach', 'buildings': 'commercial buildings', 'chaparral': 'chaparral shrubland', 'denseresidential': 'dense residential housing',
                'forest': 'a forest', 'freeway': 'a freeway', 'golfcourse': 'a golf course', 'harbor': 'a harbor with boats', 'intersection': 'a road intersection',
                'mediumresidential': 'medium density residential houses', 'mobilehomepark': 'a mobile home park', 'overpass': 'a highway overpass',
                'parkinglot': 'a parking lot with cars', 'river': 'a river', 'runway': 'an airport runway', 'sparseresidential': 'sparse residential houses',
                'storagetanks': 'industrial storage tanks', 'tenniscourt': 'tennis courts'}

TEMPLATES = ["an aerial photo of {}.", "a satellite image of {}.", "a top-down drone view of {}.", "an overhead photo of {}."]

# Operational meaning for a drone-ops agent: is this scene flight-sensitive?
FLIGHT_SENSITIVITY = {'airplane': 'restricted airspace likely — airport', 'runway': 'restricted airspace likely — airport',
                      'denseresidential': 'flight over people risk — require waiver', 'mediumresidential': 'flight over people risk — require waiver',
                      'mobilehomepark': 'flight over people risk — require waiver', 'storagetanks': 'industrial site — check exclusion zones',
                      'harbor': 'crane / mast exclusion zones', 'freeway': 'moving vehicles — no hover', 'overpass': 'moving vehicles — no hover',
                      'beach': 'crowds possible — check occupancy'}
