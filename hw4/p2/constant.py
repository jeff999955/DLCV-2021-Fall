from torchvision import transforms

tfm = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


mini_train_dir = "/tmp2/b08902134/hw4-neverloses87/hw4_data/mini/train"
mini_valid_dir = "/tmp2/b08902134/hw4-neverloses87/hw4_data/mini/val"
office_train_dir = "/tmp2/b08902134/hw4-neverloses87/hw4_data/office/train"
office_valid_dir = "/tmp2/b08902134/hw4-neverloses87/hw4_data/office/val"

office_labels = {'Flipflops': 0, 'Calculator': 1, 'Knives': 2, 'Exit_Sign': 3, 'Hammer': 4, 'Backpack': 5, 'Monitor': 6, 'Desk_Lamp': 7, 'TV': 8, 'Telephone': 9, 'Folder': 10, 'Keyboard': 11, 'Sneakers': 12, 'Flowers': 13, 'Postit_Notes': 14, 'Pencil': 15, 'Fork': 16, 'Lamp_Shade': 17, 'Alarm_Clock': 18, 'Computer': 19, 'File_Cabinet': 20, 'Calendar': 21, 'Ruler': 22, 'Pan': 23, 'Shelf': 24, 'Trash_Can': 25, 'Mop': 26, 'Webcam': 27, 'Bottle': 28, 'Sink': 29, 'Radio': 30, 'Printer': 31, 'Bed': 32, 'Oven': 33, 'Fan': 34, 'Batteries': 35, 'Drill': 36, 'Marker': 37, 'Laptop': 38, 'Helmet': 39, 'Spoon': 40, 'Notebook': 41, 'Mug': 42, 'Soda': 43, 'Table': 44, 'Toys': 45, 'Bike': 46, 'Scissors': 47, 'Push_Pin': 48, 'Screwdriver': 49, 'Eraser': 50, 'Mouse': 51, 'Curtains': 52, 'ToothBrush': 53, 'Clipboards': 54, 'Bucket': 55, 'Candles': 56, 'Glasses': 57, 'Paper_Clip': 58, 'Couch': 59, 'Chair': 60, 'Kettle': 61, 'Refrigerator': 62, 'Pen': 63, 'Speaker': 64}

office_list_labels = list(office_labels.keys())

mini_train_labels = {'n02108551': 0, 'n02747177': 1, 'n01843383': 2, 'n04243546': 3, 'n01910747': 4, 'n02091831': 5, 'n02966193': 6, 'n03062245': 7, 'n03854065': 8, 'n03400231': 9, 'n03676483': 10, 'n06794110': 11, 'n02111277': 12, 'n13054560': 13, 'n03908618': 14, 'n03017168': 15, 'n04389033': 16, 'n02105505': 17, 'n03337140': 18, 'n01770081': 19, 'n04515003': 20, 'n03527444': 21, 'n04258138': 22, 'n04275548': 23, 'n07747607': 24, 'n01532829': 25, 'n02457408': 26, 'n03347037': 27, 'n04612504': 28, 'n03888605': 29, 'n03047690': 30, 'n07584110': 31, 'n02113712': 32, 'n02101006': 33, 'n02120079': 34, 'n02606052': 35, 'n02687172': 36, 'n04296562': 37, 'n03924679': 38, 'n09246464': 39, 'n04443257': 40, 'n13133613': 41, 'n01704323': 42, 'n02089867': 43, 'n02823428': 44, 'n04435653': 45, 'n02074367': 46, 'n03207743': 47, 'n02165456': 48, 'n04067472': 49, 'n01558993': 50, 'n02108089': 51, 'n04251144': 52, 'n07697537': 53, 'n03476684': 54, 'n03838899': 55, 'n04596742': 56, 'n01749939': 57, 'n03220513': 58, 'n04604644': 59, 'n03998194': 60, 'n02108915': 61, 'n04509417': 62, 'n02795169': 63}
mini_val_labels = {'n02971356': 0, 'n03770439': 1, 'n03535780': 2, 'n02114548': 3, 'n02091244': 4, 'n03417042': 5, 'n03075370': 6, 'n02950826': 7, 'n09256479': 8, 'n03584254': 9, 'n03980874': 10, 'n02138441': 11, 'n01855672': 12, 'n02981792': 13, 'n02174001': 14, 'n03773504': 15}

mini_train_list_labels = list(mini_train_labels.keys())
mini_val_list_labels = list(mini_val_labels.keys())

if __name__ == "__main__":
    print(office_labels)
