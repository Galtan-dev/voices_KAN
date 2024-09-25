import os

path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\unhealthy"
i = 0

for item in os.listdir(path):
    # try:
    #     parts = item.split("_")
    #     if parts[1] == "healthy":
    #         source = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing" + "\\" + item
    #         destination = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\unhealthy" + "\\" + item
    #         os.rename(source, destination)
    # except Exception as ex:
    #     print(ex)
    i += 1
print(i)
