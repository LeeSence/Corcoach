from django.db import models
# Create your models here.

class coach_turtle(models.Model):
    idx = models.AutoField(primary_key=True)
    dates = models.DateTimeField()
    path = models.CharField(max_length=100)
    id = models.CharField(max_length=20)

class coach_shoulder(models.Model):
    idx = models.AutoField(primary_key=True)
    dates = models.DateTimeField()
    path = models.CharField(max_length=100)
    id = models.CharField(max_length=20)
    
class coach_nail(models.Model):
    idx = models.AutoField(primary_key=True)
    dates = models.DateTimeField()
    path = models.CharField(max_length=100)
    id = models.CharField(max_length=20)