import threading

from flask import Flask, request, redirect, url_for, send_file
import monocamera_location
import detect_mod

api = Flask(__name__)

@api.route("/SendPictureDataQuery", methods=["POST"])
def getResult():
    global image_file,detection
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
    if request.files.get("det"):
        detection = request.files["det"]
    return

@api.route("/getResult",methods=["GET"])
def returnResult():
    global image_file, detection
    if image_file is not None and detection is not None:
        return {"x":-1,"y":-1,"z":-1}
    detection
    if not request.method == "GET":
        threeDlocation = monocamera_location.calculate(image_file,detection)
        return {"x":threeDlocation[0],"y":threeDlocation[1],"z":threeDlocation[2] }
    return

def run():
    model = threading.Thread(target=detect_mod.run)
    model.start()
    api.run(host="0.0.0.0", port=80, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
