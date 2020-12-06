from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
# Create your views here.


def home(request):
    return render(request, 'home.html')


def transactions(request):
    return render(request, 'transactions.html')


def login(request):
    return render(request, 'login.html')


def report(request):
    return render(request, 'report.html')


def ml(request):
    return render(request, 'ml.html')


def signup(request):
    # Adding users

    if request.method == 'POST':
        username = request.POST['username']
        fname = request.POST['fname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        address = request.POST['address']
        city = request.POST['city']
        state = request.POST['state']
        zip = request.POST['zip']

        myuser = User.objects.create_user(username, email, pass1)
        myuser.save()
        return redirect('/login')
    else:
        return render(request, 'signup.html')
