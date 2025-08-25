import cv2
import os
import json

def click(event, x, y, flags, param):

    '''
    Simple function that allows you to store the x and y pixel coordinates in a dictionary after clicking on a certain image

    Input:
        - event: button click (left button down in this case)
        - x (int): x coordinate after clicking on the picture
        - y (int): y coordinate after clicking on the picture
        - flags (None): undefined, required for the cv2 function templates
        - param (None): undefined, required for the cv2 function templates

    Output:
        - None
    '''
    global center_pts, window_close, image_path

    if event == cv2.EVENT_LBUTTONDOWN:

        # Dictionary para depois guardar dados em json facilmente
        coordinate_dict = {
            "image_path": image_path,
            "x": x,
            "y": y
        }
        center_pts.append(coordinate_dict)
        window_close = True

if __name__ == "__main__":
    
    input_dir = "../NewRGBImages/images_with_gt" # change directory
    center_pts = []
    window_close = False # Necessário definir esta condição para poder fechar a janela sem usar .destroyAllWindows() diretamente na função (problemas de GUI no MacOS)

    images = [f for f in os.listdir(input_dir)]

    for image in images:
    
        image_path = image # global variable para guardar a key do dicionario de center points
        #print(image_path)
        window_close = False # Reiniciar a condição para cada nova imagem

        file_path = input_dir + '/' + image

        img = cv2.imread(file_path)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click)

        while True:

            cv2.imshow("image",img)
            key = cv2.waitKey(1) & 0xFF

            if window_close:
                break

            if key == ord("c"):
                break

        cv2.destroyAllWindows()
        print(f"Captured points so far: {center_pts}")

# Save the coordinates to a JSON file
with open("inesc_coordinates.json", "w") as json_file: # change file to save coordinates
    json.dump(center_pts, json_file, indent=4)

    print(f"Captured points so far: {center_pts}")


