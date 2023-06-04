hideChat(0);
$('#prime').click(function() {
  toggleFab();
});


//Toggle chat and links
function toggleFab() {
  $('.prime').toggleClass('zmdi-comment-outline');
  $('.prime').toggleClass('zmdi-close');
  $('.prime').toggleClass('is-active');
  $('.prime').toggleClass('is-visible');
  $('#prime').toggleClass('is-float');
  $('.chat').toggleClass('is-visible');
  $('.fab').toggleClass('is-visible');
  
}

  $('#chat_first_screen').click(function(e) {
        hideChat(1);
  });

  $('#chat_second_screen').click(function(e) {
        hideChat(2);
  });

  $('#chat_third_screen').click(function(e) {
        hideChat(3);
  });

  $('#chat_fourth_screen').click(function(e) {
        hideChat(4);
  });

  $('#chat_fullscreen_loader').click(function(e) {
      $('.fullscreen').toggleClass('zmdi-window-maximize');
      $('.fullscreen').toggleClass('zmdi-window-restore');
      $('.chat').toggleClass('chat_fullscreen');
      $('.fab').toggleClass('is-hide');
      $('.header_img').toggleClass('change_img');
      $('.img_container').toggleClass('change_img');
      $('.chat_header').toggleClass('chat_header2');
      $('.fab_field').toggleClass('fab_field2');
      $('.chat_converse').toggleClass('chat_converse2');
      //$('#chat_converse').css('display', 'none');
     // $('#chat_body').css('display', 'none');
     // $('#chat_form').css('display', 'none');
     // $('.chat_login').css('display', 'none');
     // $('#chat_fullscreen').css('display', 'block');
  });

function hideChat(hide) {
    switch (hide) {
      case 0:
            $('#chat_converse').css('display', 'none');
            $('#chat_body').css('display', 'none');
            $('#chat_form').css('display', 'none');
            $('.chat_login').css('display', 'block');
            $('.chat_fullscreen_loader').css('display', 'none');
             $('#chat_fullscreen').css('display', 'none');
            break;
      case 1:
            $('#chat_converse').css('display', 'block');
            $('#chat_body').css('display', 'none');
            $('#chat_form').css('display', 'none');
            $('.chat_login').css('display', 'none');
            $('.chat_fullscreen_loader').css('display', 'block');
            break;
      case 2:
            $('#chat_converse').css('display', 'none');
            $('#chat_body').css('display', 'block');
            $('#chat_form').css('display', 'none');
            $('.chat_login').css('display', 'none');
            $('.chat_fullscreen_loader').css('display', 'block');
            break;
      case 3:
            $('#chat_converse').css('display', 'none');
            $('#chat_body').css('display', 'none');
            $('#chat_form').css('display', 'block');
            $('.chat_login').css('display', 'none');
            $('.chat_fullscreen_loader').css('display', 'block');
            break;
      case 4:
            $('#chat_converse').css('display', 'none');
            $('#chat_body').css('display', 'none');
            $('#chat_form').css('display', 'none');
            $('.chat_login').css('display', 'none');
            $('.chat_fullscreen_loader').css('display', 'block');
            $('#chat_fullscreen').css('display', 'block');
            break;
    }
}

$(document).ready(function() {
var userResponseTimer; // Variable to track user response time

function getBotResponse() {
  var rawText = $('#chatSend').val();
  var userHtml = '<div class="chat_msg_item chat_msg_item_user">' + rawText + '</div>';
  $('#chatSend').val("");
  $('#chat_converse').append(userHtml);

  // Clear the previous timer if any
  clearTimeout(userResponseTimer);

  var typingHtml = '<div class="chat_msg_item chat_msg_item_admin typing"><div class="chat_avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR41Fkqg3c37MAQJPPJntb-ZFYqrzaEDmqCXh2hI7Q-dewIpmmN6TwLW1xiF_Eik-jKNOg&usqp=CAU"/></div><span class="typing-indicator"></span><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>';
  $("#chat_converse").append(typingHtml);

  $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
      // Remove the "typing..." message
      $(".typing").remove();

      for (var i = 0; i < data.length; i++) {
          var message = data[i];
          var messageHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR41Fkqg3c37MAQJPPJntb-ZFYqrzaEDmqCXh2hI7Q-dewIpmmN6TwLW1xiF_Eik-jKNOg&usqp=CAU"/></div>' + message + '</span>';
          $('#chat_converse').append(messageHtml);
          scrollToBottom();
      }

      document.getElementById('userInput').scrollIntoView({ block: 'start', behavior: 'smooth' });
      
  });
  
  // Set a timer to check for delayed user response
  userResponseTimer = setTimeout(function() {
      // Remove the "typing..." message
      $(".typing").remove();

      // Display a message asking for another help
      var messageHtml = '<span class="chat_msg_item chat_msg_item_admin">&#x1F62A;&#x1F62A;&#x1F62A;&#x1F62A;&#x1F62A;</span>';
      $('#chat_converse').append(messageHtml);

      document.getElementById('userInput').scrollIntoView({ block: 'start', behavior: 'smooth' });
  }, 90000); // 90 seconds
}






               

                $('#chatSend').keypress(function(e) {
                    if(e.which == 13) {
                        getBotResponse();
                        scrollToBottom();
                    }
                });
                $('#fab_send').click(function() {
                    getBotResponse();
                })
            });  
    
    
// Function to scroll to the bottom of the chat
function scrollToBottom() {
var chatConverse = document.getElementById('chat_converse');
chatConverse.scrollTop = chatConverse.scrollHeight;
}

