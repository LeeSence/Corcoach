from django.urls import path

from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
   path('', views.index, name='index'),
   path('login/', views.login, name='login'),
   path('about/', views.about, name='about'),
   path('join/', views.join, name='join'),
   path('ar_graph/', views.ar_graph, name='ar_graph'),
   path('forgot/', views.forgot, name='forgot'),
   path('save_pic/', views.save_pic, name='save_pic'),
   path('nail_pic/', views.nail_pic, name='nail_pic'),
   path('shoulder_pic/', views.shoulder_pic, name='shoulder_pic'),
   path('turtle_pic/', views.turtle_pic, name='turtle_pic'),
   path('update/', views.update, name='update'),
   path('start/', views.start, name='start'),
   path('signup/', views.signup, name='signup'),
   path('logout/', views.logout, name='logout'),
   path('up_image/', views.up_image, name='up_image'),
   path('video_view/', views.video_view, name='video_view'),
   path('turtle/', views.turtle, name='turtle'),
   path('nail/', views.nail, name='nail'),
   path('shoulder/', views.shoulder, name='shoulder'),
   #path('leg_cross/', views.leg_cross, name='leg_cross'),
   path('func_turtle/', views.func_turtle, name='func_turtle'),
   path('func_nail/', views.func_nail, name='func_nail'),
   path('func_shoulder/', views.func_shoulder, name='func_shoulder'),
   path('func_leg_cross/', views.func_leg_cross, name='func_leg_cross'),
   path('updatePassword/', views.change_password, name='updatePassword'),
   path('updateCheck/', views.updateCheck, name='updateCheck'),
   path('passwordCheck/', views.passwordCheck, name='passwordCheck'),
]
