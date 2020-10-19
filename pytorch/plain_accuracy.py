def plain_acc(net, by_class = 0):
    if by_class:
        class_correct = [0.0 for i in range(10)]
        class_total = [0.0 for i in range(10)]

        #boşuna grad tensoru açmaması ve fonksiyonları takip ile uğraşmaması için no_grad
        with torch.no_grad():
            for data in testloader: # any iterable dataset can be written here instead of testloader
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %% num : %6d' % (
                classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))
        return class_correct, class_total
        
    else:
        correct = 0
        total = 0

        #gradlar için ayrıdan matris açmasın diye with no grad altında
        with torch.no_grad():
            for data in testloader: # any iterable dataset can be written here instead of testloader
                images,labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))   
        correct_test = correct

        with torch.no_grad():
            for i,data in enumerate(trainloader): # any iterable dataset can be written here instead of testloader
                images,labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # till 4*2500 images reached
                if i == 2500: break

        print('Accuracy of the network on the 10000 trainset images: %d %%' % (
            100 * correct / total))
        correct_train = correct
        return correct_test, correct_train
