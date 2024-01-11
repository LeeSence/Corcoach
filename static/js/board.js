$.ajax({
  type: "POST",
  url: "/enfpapp/start/",
  data: $(this).val(),
  success: function (response) {
    console.log($(this).val());
  },
});
