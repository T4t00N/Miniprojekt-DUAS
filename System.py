import os
import cv2 as cv
import numpy as np
import joblib



#   Configuration and path setup:
path_to_data = r'C:\Users\anto3\Desktop\DAKI\miniprojekt-DUAS-main'
classifier_path = os.path.join(path_to_data, 'RF_classifier.joblib')
Label_encoder_path = os.path.join(path_to_data, 'label_encoder.joblib')
Image_dir = os.path.join(path_to_data, 'Data', 'Cropped and perspective corrected boards')
templates_path = [os.path.join(path_to_data, 'Data', 'Crowns_martin', f'{i}_DS.png') for i in range(1,4)]

#   Key values: (0.75)
Threshold = 0.75

#   Load classifier and label encoder.
clf = joblib.load(classifier_path)
label_encoder = joblib.load(Label_encoder_path)


#   Loads the board images
def load_image(path):
    """Load the image from the given path."""
    image = cv.imread(path)
    if image is None:
        print(f"Failed to load image at {path}")
    return image

#   Load the templates
def load_templates(template_paths):
    """Load and rotate templates from the given paths."""

    templates = []
    for path in template_paths:
        image = load_image(path)
        if image is not None:
            #   Rotate the templates to make a template for every orthogonal direction.
            templates.extend([cv.rotate(image, rot) for rot in (cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE)])
            templates.append(image)
    return templates


def match_templates(tile, templates, Threshold):
    """Match templates in a given tile, return match count and locations."""
    match_locations = []
    scores = []
    boxes = []
    #   Check the picture if the templates fir anywhere, and maches at or more that the given threshold.
    for template in templates:
        res = cv.matchTemplate(tile, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= Threshold)
        #   Make red box on the places a match is found.
        for pt in zip(*loc[::-1]):  # switch x and y locations
            w, h = template.shape[1], template.shape[0]
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
            scores.append(res[pt[1], pt[0]])

    boxes = np.array(boxes)
    scores = np.array(scores)
    if boxes.any():
        boxes = non_max_suppression(boxes, scores, 0.3)

    for (startX, startY, endX, endY) in boxes:
        match_locations.append((startX, startY))

    return len(boxes), match_locations

def non_max_suppression(boxes, scores, overlapThresh):
    """Perform non-maximum suppression given boxes and scores."""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def predict_terrain(hue, saturation, value):
    """Predict the terrain type based on hue, saturation, and value."""
    features = np.array([[hue, saturation, value]])
    terrain_encoded = clf.predict(features)
    terrain = label_encoder.inverse_transform(terrain_encoded)[0]
    return terrain

def get_tiles(image, templates):
    """Extract tiles from an image, analyze them for terrain and template matches, and print detailed outputs."""
    board_size = 100  # Assuming each tile is 100x100 pixels
    num_tiles_per_row = image.shape[1] // board_size
    num_tiles_per_col = image.shape[0] // board_size

    tiles = []
    for y in range(num_tiles_per_col):
        row = []
        for x in range(num_tiles_per_row):
            tile = image[y*board_size:(y+1)*board_size, x*board_size:(x+1)*board_size]
            hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
            hue, saturation, value = np.median(hsv_tile, axis=(0, 1))
            terrain = predict_terrain(hue, saturation, value)
            template_matches, match_locations = match_templates(tile, templates, Threshold)

            #   Print detailed information about each tile
            print(f"Tile [{y}, {x}] - Match Locations: {match_locations}")

            row.append({
                'x': x, 'y': y, 'terrain': terrain,
                'template_matches': template_matches, 'match_locations': match_locations
            })
        tiles.append(row)
    return tiles

def process_image(image_path, templates):
    """Process a single image for tile analysis, territory calculation, and visualization, and print results."""
    image = load_image(image_path)
    if image is None:
        return

    tiles = get_tiles(image, templates)
    territories = calculate_territories(tiles)

    # Calculate and print the score
    score = sum(territory_size * matches for _, (territory_size, matches) in territories)
    print("Territories:", territories)
    print(f"The score is {score}!")

    visualize_matches(image.copy(), tiles, image_path)



def calculate_territories(tiles):
    """Calculate and return territories from analyzed tiles."""
    visited = [[False for _ in range(len(tiles[0]))] for _ in range(len(tiles))]
    territories = []

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):
            if not visited[y][x]:
                terrain_type = tiles[y][x]['terrain']
                territory_size, matches = dfs(x, y, visited, tiles, terrain_type)
                territories.append((terrain_type, (territory_size, matches)))
    return territories

def dfs(x, y, visited, tiles, terrain_type):
    """Depth-First Search to explore territory extent."""
    if x < 0 or x >= len(tiles[0]) or y < 0 or y >= len(tiles) or visited[y][x] or tiles[y][x]['terrain'] != terrain_type:
        return 0, 0
    visited[y][x] = True
    size = 1
    matches = tiles[y][x]['template_matches']

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        s, m = dfs(nx, ny, visited, tiles, terrain_type)
        size += s
        matches += m
    return size, matches

def visualize_matches(image, tiles, image_path):
    """Visualize match locations on an image."""
    

    for row in tiles:
        for tile in row:
            for (loc_x, loc_y) in tile['match_locations']:
                global_x = tile['x'] * 100 + loc_x
                global_y = tile['y'] * 100 + loc_y
                box_width = 20
                box_height = 20
                cv.rectangle(image, (global_x, global_y), (global_x + box_width, global_y + box_height), (0, 0, 255), -1)


    
    image_name = os.path.basename(image_path)  # Get the image name from the path, as a window header.
    cv.imshow(image_name, image)  # Display the new image with squares on the detected crowns.
    cv.waitKey(0)   #   Show one picture at the time(Outcomment to make it run through all pics fast.)
    cv.destroyAllWindows()  #   Close all windows

# Main function to execute it all.
def main():
    # Load the templates
    templates = load_templates(templates_path)
    print(f"Templates loaded from {templates_path}")

    for i in range(1,75):  # Process all 75 images
        # Path to image
        image_path = os.path.join(Image_dir, f"{i}.jpg")
        # If the pics can't be found.
        if not os.path.isfile(image_path):
            print(f"Image not found: {image_path}")
            continue
        process_image(image_path, templates)


# Ensure this part remains unchanged
if __name__ == "__main__":
    main()